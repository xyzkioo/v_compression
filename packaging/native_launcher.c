/* This project's folder launcher: resolves itself after a folder move. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

static int fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    if (getenv("DISPLAY") || getenv("WAYLAND_DISPLAY")) {
        pid_t dialog = fork();
        if (dialog == 0) {
            execlp("zenity", "zenity", "--error", "--title=v_compression 启动失败",
                   "--text", message, (char *)NULL);
            _exit(127);
        }
        if (dialog > 0) {
            while (waitpid(dialog, NULL, 0) < 0 && errno == EINTR) {}
        }
    }
    return 127;
}

int main(int argc, char **argv) {
    char executable[PATH_MAX];
    ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (size < 0 || (size_t)size >= sizeof(executable) - 1)
        return fail("无法确定本程序所在目录。请保留完整项目文件夹。");
    executable[size] = '\0';
    char *separator = strrchr(executable, '/');
    if (!separator) return fail("无法确定项目目录。");
    *separator = '\0';
    char script[PATH_MAX];
    int written = snprintf(script, sizeof(script), "%s/launcher.py", executable);
    if (written < 0 || (size_t)written >= sizeof(script))
        return fail("项目目录路径过长。");
    if (access(script, R_OK) != 0)
        return fail("找不到同一文件夹内的 launcher.py。请移动整个项目文件夹，而非单独移动启动文件。");
    char **arguments = calloc((size_t)argc + 4, sizeof(char *));
    if (!arguments) return fail("启动器内存分配失败。");
    arguments[0] = "python3";
    arguments[1] = script;
    int next = 2;
    if (argc == 1) arguments[next++] = "--desktop-launch";
    for (int index = 1; index < argc; ++index) arguments[next++] = argv[index];
    arguments[next] = NULL;
    execvp(arguments[0], arguments);
    free(arguments);
    return fail("找不到可运行的 python3。请安装 Python 3 后重新打开。");
}
