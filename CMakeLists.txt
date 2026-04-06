cmake_minimum_required(VERSION 4.0)
project(Projet_Annuel_3IABD C)

set(CMAKE_C_STANDARD 11)

add_library(Projet_Annuel_3IABD SHARED
        pmc/lib/lib.c
        pmc/lib/lib.h
        pmc/utils/pmc_utils.c
        pmc/struct/pmc_struct.h
        pmc/utils/pmc_utils.h
)