# SPDX-License-Identifier: Apache-2.0
# RGP-R97: tiny user-mode debugger (ctypes, no system change). Runs a command under DEBUG_ONLY_THIS_PROCESS,
# logs DLL loads (path+base), every exception (code, address -> module+offset, first/second chance) and the exit code.
# usage: python minidbg.py LOGFILE -- cmd args...   (env passes through; stdout/stderr inherited)
import ctypes, sys, time, os
from ctypes import wintypes as W
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
DEBUG_ONLY_THIS_PROCESS = 2
CREATE_NEW_PROCESS_GROUP = 0x200
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR), ("lpTitle", W.LPWSTR), ("dwX", W.DWORD), ("dwY", W.DWORD),
                ("dwXSize", W.DWORD), ("dwYSize", W.DWORD), ("dwXCountChars", W.DWORD), ("dwYCountChars", W.DWORD), ("dwFillAttribute", W.DWORD),
                ("dwFlags", W.DWORD), ("wShowWindow", W.WORD), ("cbReserved2", W.WORD), ("lpReserved2", ctypes.c_void_p), ("hStdInput", W.HANDLE),
                ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]
class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE), ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]
class EXCEPTION_RECORD(ctypes.Structure):
    pass
EXCEPTION_RECORD._fields_ = [("ExceptionCode", W.DWORD), ("ExceptionFlags", W.DWORD), ("ExceptionRecord", ctypes.c_void_p), ("ExceptionAddress", ctypes.c_void_p),
                             ("NumberParameters", W.DWORD), ("ExceptionInformation", ctypes.c_uint64 * 15)]
class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("ExceptionRecord", EXCEPTION_RECORD), ("dwFirstChance", W.DWORD)]
class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hFile", W.HANDLE), ("lpBaseOfDll", ctypes.c_void_p), ("dwDebugInfoFileOffset", W.DWORD), ("nDebugInfoSize", W.DWORD),
                ("lpImageName", ctypes.c_void_p), ("fUnicode", W.WORD)]
class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hFile", W.HANDLE), ("hProcess", W.HANDLE), ("hThread", W.HANDLE), ("lpBaseOfImage", ctypes.c_void_p), ("dwDebugInfoFileOffset", W.DWORD),
                ("nDebugInfoSize", W.DWORD), ("lpThreadLocalBase", ctypes.c_void_p), ("lpStartAddress", ctypes.c_void_p), ("lpImageName", ctypes.c_void_p), ("fUnicode", W.WORD)]
class EXIT_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", W.DWORD)]
class OUTPUT_DEBUG_STRING_INFO(ctypes.Structure):
    _fields_ = [("lpDebugStringData", ctypes.c_void_p), ("fUnicode", W.WORD), ("nDebugStringLength", W.WORD)]
class U(ctypes.Union):
    _fields_ = [("Exception", EXCEPTION_DEBUG_INFO), ("LoadDll", LOAD_DLL_DEBUG_INFO), ("CreateProcess", CREATE_PROCESS_DEBUG_INFO),
                ("ExitProcess", EXIT_PROCESS_DEBUG_INFO), ("DebugString", OUTPUT_DEBUG_STRING_INFO), ("pad", ctypes.c_byte * 160)]
class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [("dwDebugEventCode", W.DWORD), ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD), ("u", U)]
k32.GetFinalPathNameByHandleW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
k32.ReadProcessMemory.argtypes = [W.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]

dbghelp = ctypes.WinDLL("dbghelp", use_last_error=True)
class ADDRESS64(ctypes.Structure):
    _fields_ = [("Offset", ctypes.c_uint64), ("Segment", ctypes.c_uint16), ("Mode", ctypes.c_uint32)]
class STACKFRAME64(ctypes.Structure):
    _fields_ = [("AddrPC", ADDRESS64), ("AddrReturn", ADDRESS64), ("AddrFrame", ADDRESS64), ("AddrStack", ADDRESS64), ("AddrBStore", ADDRESS64),
                ("FuncTableEntry", ctypes.c_void_p), ("Params", ctypes.c_uint64 * 4), ("Far", W.BOOL), ("Virtual", W.BOOL),
                ("Reserved", ctypes.c_uint64 * 3), ("KdHelp", ctypes.c_byte * 256)]
dbghelp.SymInitializeW.argtypes = [W.HANDLE, W.LPCWSTR, W.BOOL]
dbghelp.SymFunctionTableAccess64.restype = ctypes.c_void_p
dbghelp.SymFunctionTableAccess64.argtypes = [W.HANDLE, ctypes.c_uint64]
dbghelp.SymGetModuleBase64.restype = ctypes.c_uint64
dbghelp.SymGetModuleBase64.argtypes = [W.HANDLE, ctypes.c_uint64]
FTA = ctypes.WINFUNCTYPE(ctypes.c_void_p, W.HANDLE, ctypes.c_uint64)(lambda h, a: dbghelp.SymFunctionTableAccess64(h, a))
GMB = ctypes.WINFUNCTYPE(ctypes.c_uint64, W.HANDLE, ctypes.c_uint64)(lambda h, a: dbghelp.SymGetModuleBase64(h, a))
dbghelp.StackWalk64.argtypes = [W.DWORD, W.HANDLE, W.HANDLE, ctypes.POINTER(STACKFRAME64), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
k32.OpenThread.restype = W.HANDLE
k32.GetThreadContext.argtypes = [W.HANDLE, ctypes.c_void_p]
REGS = ["Rax", "Rcx", "Rdx", "Rbx", "Rsp", "Rbp", "Rsi", "Rdi", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "Rip"]
def stack(hproc, tid, where, L):
    raw = ctypes.create_string_buffer(1232 + 16)
    addr = (ctypes.addressof(raw) + 15) & ~15
    ctypes.c_uint32.from_address(addr + 0x30).value = 0x10000B
    ht = k32.OpenThread(0x001F03FF, False, tid)
    if not k32.GetThreadContext(ht, ctypes.c_void_p(addr)):
        L(f"GetThreadContext failed {ctypes.get_last_error()}"); return
    r = {n: ctypes.c_uint64.from_address(addr + 0x78 + 8 * i).value for i, n in enumerate(REGS)}
    L("REGS " + " ".join(f"{n}={v:X}" for n, v in r.items()))
    sf = STACKFRAME64()
    sf.AddrPC.Offset, sf.AddrPC.Mode = r["Rip"], 3
    sf.AddrFrame.Offset, sf.AddrFrame.Mode = r["Rbp"], 3
    sf.AddrStack.Offset, sf.AddrStack.Mode = r["Rsp"], 3
    for i in range(48):
        if not dbghelp.StackWalk64(0x8664, hproc, ht, ctypes.byref(sf), ctypes.c_void_p(addr), None, FTA, GMB, None):
            break
        L(f"  #{i:02d} 0x{sf.AddrPC.Offset:X} {where(sf.AddrPC.Offset)}")
    k32.CloseHandle(ht)

def path_of(h):
    b = ctypes.create_unicode_buffer(1024)
    n = k32.GetFinalPathNameByHandleW(h, b, 1024, 0)
    return b.value.replace("\\?\\", "") if n else "?"
def modsize(hproc, base):
    hdr = (ctypes.c_ubyte * 4096)(); got = ctypes.c_size_t()
    if not k32.ReadProcessMemory(hproc, ctypes.c_void_p(base), hdr, 4096, ctypes.byref(got)):
        return 0
    b = bytes(hdr); e = int.from_bytes(b[0x3c:0x40], "little")
    return int.from_bytes(b[e + 24 + 56:e + 24 + 60], "little")  # OptionalHeader.SizeOfImage
dbghelp.SymLoadModuleExW.argtypes = [W.HANDLE, W.HANDLE, W.LPCWSTR, W.LPCWSTR, ctypes.c_uint64, W.DWORD, ctypes.c_void_p, W.DWORD]
dbghelp.SymLoadModuleExW.restype = ctypes.c_uint64
def main():
    logf = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
    log = open(logf, "w", buffering=1)
    def L(s):
        log.write(time.strftime("%H:%M:%S ") + s + "\n")
    si = STARTUPINFO(); si.cb = ctypes.sizeof(si); pi = PROCESS_INFORMATION()
    cl = " ".join('"%s"' % c if " " in c else c for c in cmd)
    if not k32.CreateProcessW(None, ctypes.create_unicode_buffer(cl), None, None, True, DEBUG_ONLY_THIS_PROCESS | CREATE_NEW_PROCESS_GROUP, None, None, ctypes.byref(si), ctypes.byref(pi)):
        L(f"CreateProcess failed {ctypes.get_last_error()}"); return 1
    dbghelp.SymInitializeW(pi.hProcess, None, False)
    L(f"started pid {pi.dwProcessId}: {cl}")
    open(logf + ".pid", "w").write(str(pi.dwProcessId))
    mods = []; ev = DEBUG_EVENT(); nexc = 0; seen_bp = False; rc = None
    def where(a):
        for base, size, p in mods:
            if base <= a < base + max(size, 1):
                return f"{os.path.basename(p)}+0x{a - base:X}"
        return "?"
    while True:
        if not k32.WaitForDebugEvent(ctypes.byref(ev), 0xFFFFFFFF):
            L("WaitForDebugEvent failed"); break
        cont = DBG_CONTINUE; c = ev.dwDebugEventCode
        if c == 3:  # CREATE_PROCESS
            cp = ev.u.CreateProcess; p = path_of(cp.hFile); mods.append((cp.lpBaseOfImage, modsize(pi.hProcess, cp.lpBaseOfImage), p))
            dbghelp.SymLoadModuleExW(pi.hProcess, None, p, None, ctypes.c_uint64(cp.lpBaseOfImage), mods[-1][1], None, 0)
        elif c == 6:  # LOAD_DLL
            ld = ev.u.LoadDll; p = path_of(ld.hFile); base = ld.lpBaseOfDll or 0
            mods.append((base, modsize(pi.hProcess, base), p))
            dbghelp.SymLoadModuleExW(pi.hProcess, None, p, None, ctypes.c_uint64(base), mods[-1][1], None, 0)
            if any(s in p.lower() for s in ("amd", "hip", "devdriver", "rdp", "amdxc", "amdihk", "comgr")):
                L(f"LOAD {p} @0x{base:X}")
            if ld.hFile: k32.CloseHandle(ld.hFile)
        elif c == 1:  # EXCEPTION
            er = ev.u.Exception.ExceptionRecord; code = er.ExceptionCode; a = er.ExceptionAddress or 0; fc = ev.u.Exception.dwFirstChance
            if code == 0x80000003 and not seen_bp:
                seen_bp = True
            else:
                nexc += 1
                if code not in (0x406D1388, 0x40010006, 0x4001000A) and nexc < 500:  # thread-name / debug-string noise
                    info = " ".join(f"0x{er.ExceptionInformation[i]:X}" for i in range(min(er.NumberParameters, 4)))
                    L(f"EXC tid {ev.dwThreadId} code 0x{code:08X} at 0x{a:X} {where(a)} first={fc} params [{info}]")
                    if fc and code in (0xC0000005, 0xC0000409, 0xC000001D, 0xC0000374, 0xE06D7363):
                        stack(pi.hProcess, ev.dwThreadId, where, L)
                cont = DBG_EXCEPTION_NOT_HANDLED
        elif c == 5:  # EXIT_PROCESS
            rc = ev.u.ExitProcess.dwExitCode
            L(f"EXIT code {rc} (0x{rc:08X})")
            k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, cont); break
        k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, cont)
    return 0 if rc == 0 else 3
sys.exit(main())
