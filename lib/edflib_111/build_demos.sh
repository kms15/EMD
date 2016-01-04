#!/bin/sh
#
gcc main.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -o test_edflib
#
gcc sine.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o sine
#
gcc test_generator.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o testgenerator
#
gcc sweep.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o sweep
#
