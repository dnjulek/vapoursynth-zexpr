/* Plugin-only Windows DLL entry point for explicit FreeLibrary teardown.
 * Compiled without dllexport so the PE export table keeps only VapourSynthPluginInit2.
 */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

extern void zj_run_dso_teardown_now(void);

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst;
    if (reason == DLL_PROCESS_DETACH && reserved == NULL) {
        zj_run_dso_teardown_now();
    }
    return TRUE;
}
#endif
