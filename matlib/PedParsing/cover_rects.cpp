#include "mex.h"
#include <vector>
#include <set>
#include <algorithm>

using std::vector;

// function [cover_rects] = cover_rects(mask, margin)

struct Rect
{
    int x1;
    int y1;
    int x2;
    int y2;

    Rect()
        : x1(),
          y1(),
          x2(),
          y2()
    {

    }

    Rect(int x1, int y1, int x2, int y2)
        : x1(x1),
          y1(y1),
          x2(x2),
          y2(y2)
    {

    }

    Rect& inflate(const Rect& another)
    {
        x1 = std::min(x1, another.x1);
        y1 = std::min(y1, another.y1);
        x2 = std::max(x2, another.x2);
        y2 = std::max(y2, another.y2);
        return *this;
    }
};

class DisjointSets
{
public:
    typedef size_t size_type;

public:
    explicit DisjointSets(size_type n) 
    : _nsets(n)
    {
        _parents.reserve(n);
        _ranks.reserve(n);
        for (size_type i = 0; i < n; ++i) 
        {
            _parents.push_back(i);
            _ranks.push_back(0);
        }
    }

    // basic info

    size_type size() const { return _parents.size(); }

    size_type nsets() const { return _nsets; }

    bool singleton() const { return _nsets <= 1; }

    // operations

    size_type root(size_type x) const
    {
        size_type p = _parents[x];
        return x == p ? x : (_parents[x] = root(p));
    }

    std::vector<size_type> roots() const
    {
        std::set<size_type> rootSet;
        for (size_type i = 0; i < _parents.size(); ++i) {
            size_type ri = root(i);
            rootSet.insert(ri);
        }

        std::vector<size_type> ret;
        for (std::set<size_type>::const_iterator it = rootSet.begin(); 
                it != rootSet.end(); ++it) {
            ret.push_back(*it);
        }

        return ret;
    }

    size_type link(size_type x, size_type y)
    {
        size_type px = root(x);
        size_type py = root(y);
        if (px == py) return px;

        size_type rx = _ranks[px];
        size_type ry = _ranks[py];
        _nsets --;

        if (rx < ry)
        {
            _parents[px] = py;
            return py;
        }
        else
        {
            _parents[py] = px;
            if (rx == ry) _ranks[px] ++;
            return px;
        }
    }

private:
    mutable std::vector<size_type> _parents;
    std::vector<size_type> _ranks;
    size_type _nsets;

}; // class DisjointSets

bool isRectOverlap(const Rect& a, const Rect& b)
{
    int x1 = std::max(a.x1, b.x1);
    int y1 = std::max(a.y1, b.y1);
    int x2 = std::min(a.x2, b.x2);
    int y2 = std::min(a.y2, b.y2);
    return x1 <= x2 && y1 <= y2;
}

vector<Rect> getCover(const bool* mask, size_t m, size_t n, int margin)
{
    vector<Rect> all;
    for (int x = 0; x < static_cast<int>(n); ++x)
        for (int y = 0; y < static_cast<int>(m); ++y) {
            if (mask[y + x*m]) {
                int x1 = std::max(0, x-margin);
                int y1 = std::max(0, y-margin);
                int x2 = std::min(static_cast<int>(n)-1, x+margin);
                int y2 = std::min(static_cast<int>(m)-1, y+margin);
                all.push_back(Rect(x1, y1, x2, y2));
            }
        }

    DisjointSets dsets(all.size());
    for (vector<Rect>::size_type i = 0; i < all.size(); ++i)
        for (vector<Rect>::size_type j = i+1; j < all.size(); ++j) {
            DisjointSets::size_type ri = dsets.root(i);
            DisjointSets::size_type rj = dsets.root(j);
            if (ri != rj && isRectOverlap(all[ri], all[rj])) {
                DisjointSets::size_type f = dsets.link(ri, rj);
                if (ri == f) {
                    all[ri].inflate(all[rj]);
                } else {
                    all[rj].inflate(all[ri]);
                }
            }
        }

    vector<DisjointSets::size_type> roots = dsets.roots();

    vector<Rect> rects(roots.size());
    for (vector<Rect>::size_type i = 0; i < rects.size(); ++i) {
        rects[i] = all[ roots[i] ];
    }

    return rects;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 2) {
        mexErrMsgTxt("Invalid number of inputs");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Invalid number of outputs");
    }

    if (mxGetNumberOfDimensions(prhs[0]) != 2 ||
        mxGetClassID(prhs[0]) != mxLOGICAL_CLASS) {
        mexErrMsgTxt("Input image must be 2D logical");
    }
    bool* mask = static_cast<bool*>(mxGetLogicals(prhs[0]));
    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);

    int margin = static_cast<int>(mxGetScalar(prhs[1]));
    if (margin <= 0) {
        mexErrMsgTxt("Input margin must be greater than zero");
    }

    vector<Rect> rects = getCover(mask, m, n, margin);

    plhs[0] = mxCreateNumericMatrix(4, rects.size(), mxINT32_CLASS, mxREAL);
    int* ret = static_cast<int*>(mxGetData(plhs[0]));
    for (vector<Rect>::size_type i = 0; i < rects.size(); ++i) {
        ret[i*4] = rects[i].x1 + 1;
        ret[i*4+1] = rects[i].y1 + 1;
        ret[i*4+2] = rects[i].x2 + 1;
        ret[i*4+3] = rects[i].y2 + 1;
    }
}
