#pragma once

#include "plog/Log.h"
#include "plog/Logger.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>

using namespace std;

namespace rasbery {
static const int PLUS  = 1;
static const int MINUS = -1;

static const int XDIR    = 0;
static const int YDIR    = 1;
static const int ZDIR    = 2;
static const int NDIRMAX = 3;

static const int LEFT      = 0;
static const int RIGHT     = 1;
static const int LEFTRIGHT = 2;
static const int LR        = 2;
static const int CENTER    = 2;
static const int LRC       = 3;

static const int UP      = 0;
static const int DOWN    = 1;
static const int NUPDOWN = 2;

static const int FORWARD  = 0;
static const int BACKWARD = 1;
static const int NFORBACK = 2;

static const int POSITIVE = 0;
static const int NEGATIVE = 1;
static const int NSLOPE   = 2;

static const int SELF     = 0;
static const int NEIB     = 1;
static const int SELFNEIB = 2;

static const int NW = 0;
static const int NE = 1;
static const int SW = 2;
static const int SE = 3;

static const int WEST          = 0;
static const int EAST          = 1;
static const int NORTH         = 2;
static const int SOUTH         = 3;
static const int BOT           = 4;
static const int TOP           = 5;
static const int NEWS          = 4;
static const int NEWSBT        = 6;
static const int NEWS2XY[NEWS] = {XDIR, XDIR, YDIR, YDIR};

static const int    NDIVREG = 8;
static const double RDIVREG = 1.0 / NDIVREG;

static const double PI  = 3.14159265359;
static const double BIG = 1.E+300;
static const double EPS = 1.0E-10;

static const double MICRO  = 1.E-6;
static const double MILLI  = 1.E-3;
static const double KELVIN = 273.15;

static const int NG2 = 2;

static const int NPTM = 3;

// Atomic/molecular weights and Avogadro's number for the number-density relation
// N = rho * AVOG / AW.  Held as float they carried only ~7 significant digits into what is
// otherwise an all-double chain, so any future caller would inherit a ~1e-7 relative bias on
// every boron / moderator number density -- systematic, not round-off, because the same
// rounded literal is reused for every node.  Verified: as of this commit none of the six has
// a call site anywhere in the tree, so widening them changes no result today.  They are kept
// (rather than deleted) because they are the declared constants for a relation the solver
// still needs; this makes them safe to pick up.
static const double HAW   = 1.0079;
static const double OAW   = 15.994915;
static const double H2OAW = 18.010715;

static const double AVOG  = 0.6022045;
static const double B10AW = 10.012937;
static const double B11AW = 11.009305;

static const int TF_POINT = 20;

static const double sq2  = 1.414213562373;
static const double rsq2 = 0.707106781186;

const double xi[8] = {-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
                      0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565};
const double wi[8] = {0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                      0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363};

const double xi4[4] = {-0.8611363116, -0.3399810436,
                       0.3399810436, 0.8611363116};
const double wi4[4] = {0.3478548451, 0.6521451549,
                       0.6521451549, 0.3478548451};

// XSTYPE enum moved to Chiffon::XSTYPE in Model.h

enum BranchType {
    REFR,
    BPPM,
    TFUL,
    DMOD,
    TMOD
};
}
