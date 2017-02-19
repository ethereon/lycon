#include "lycon/util/file.h"

#include <cstdlib>
#include <unistd.h>

namespace lycon
{

String tempfile(const char* suffix)
{
    String fname;
    const char* temp_dir = getenv("LYCON_TEMP_PATH");
    char defaultTemplate[] = "/tmp/__lycon_temp.XXXXXX";
    if (temp_dir == 0 || temp_dir[0] == 0)
    {
        fname = defaultTemplate;
    }
    else
    {
        fname = temp_dir;
        char ech = fname[fname.size() - 1];
        if (ech != '/' && ech != '\\')
            fname = fname + "/";
        fname = fname + "__lycon_temp.XXXXXX";
    }
    const int fd = mkstemp((char*)fname.c_str());
    if (fd == -1)
        return String();

    close(fd);
    remove(fname.c_str());
    if (suffix)
    {
        if (suffix[0] != '.')
            return fname + "." + suffix;
        else
            return fname + suffix;
    }
    return fname;
}
}
