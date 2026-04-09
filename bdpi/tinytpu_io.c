/*
 * tinytpu_io.c — BDPI C helpers for TbTinyTPURuntime.bsv
 *
 * tinytpu_bundle_open():    open bundle file given by $TINYTPU_BUNDLE env var
 * tinytpu_bundle_read_int(): read next whitespace-delimited integer from bundle
 *
 * Both return -1 on error.  tinytpu_bundle_read_int() returns -999 on EOF.
 */

#include <stdio.h>
#include <stdlib.h>

static FILE *bundle_fp = NULL;

int tinytpu_bundle_open(void)
{
    const char *path = getenv("TINYTPU_BUNDLE");
    if (!path) {
        fprintf(stderr, "[tinytpu_io] TINYTPU_BUNDLE env var not set\n");
        return -1;
    }
    bundle_fp = fopen(path, "r");
    if (!bundle_fp) {
        fprintf(stderr, "[tinytpu_io] cannot open bundle file: %s\n", path);
        return -1;
    }
    return 0;
}

int tinytpu_bundle_read_int(void)
{
    if (!bundle_fp) return -1;
    int v = 0;
    /* skip '#' comment lines */
    int ch;
    while (1) {
        /* skip whitespace */
        while ((ch = fgetc(bundle_fp)) != EOF && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'))
            ;
        if (ch == EOF) return -999;
        if (ch == '#') {
            /* skip to end of line */
            while ((ch = fgetc(bundle_fp)) != EOF && ch != '\n')
                ;
            continue;
        }
        /* put back the non-whitespace, non-comment char and read as int */
        ungetc(ch, bundle_fp);
        break;
    }
    if (fscanf(bundle_fp, "%d", &v) != 1) return -999;
    return v;
}
