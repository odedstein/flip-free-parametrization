
#include "quartic_polynomial.h"

#include <limits>
#include <cmath>
#include <algorithm>


// Code in this file is based on poly34.h and poly34.cpp
// (c) Khashin S.I. http://math.ivanovo.ac.ru/dalgebra/Khashin/index.html
// khash2 (at) gmail.com
// Thanks to Alexandr Rakhmanin <rakhmanin (at) gmail.com>
// public domain

// x - array of size 2
// return 2: 2 real roots x[0], x[1]
// return 0: pair of complex roots: x[0]±i*x[1]
template <typename Scalar>
static int   SolveP2(Scalar *x, Scalar a, Scalar b);            // solve equation x^2 + a*x + b = 0

// x - array of size 3
// return 3: 3 real roots x[0], x[1], x[2]
// return 1: 1 real root x[0] and pair of complex roots: x[1]±i*x[2]
template <typename Scalar>
static int   SolveP3(Scalar *x, Scalar a, Scalar b, Scalar c);            // solve cubic equation x^3 + a*x^2 + b*x + c = 0

// x - array of size 4
// return 4: 4 real roots x[0], x[1], x[2], x[3], possible multiple roots
// return 2: 2 real roots x[0], x[1] and complex x[2]±i*x[3],
// return 0: two pair of complex roots: x[0]±i*x[1],  x[2]±i*x[3],
template <typename Scalar>
static int   SolveP4(Scalar *x,Scalar a,Scalar b,Scalar c,Scalar d);    // solve equation x^4 + a*x^3 + b*x^2 + c*x + d = 0 by Dekart-Euler method

// x - array of size 5
// return 5: 5 real roots x[0], x[1], x[2], x[3], x[4], possible multiple roots
// return 3: 3 real roots x[0], x[1], x[2] and complex x[3]±i*x[4],
// return 1: 1 real root x[0] and two pair of complex roots: x[1]±i*x[2],  x[3]±i*x[4],
template <typename Scalar>
static int   SolveP5(Scalar *x, Scalar a, Scalar b, Scalar c, Scalar d, Scalar e);    // solve equation x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0



template <typename Scalar>
Scalar
parametrization::quartic_polynomial(const Scalar a,
                                    const Scalar b,
                                    const Scalar c,
                                    const Scalar d)
{
    const Scalar tol = 0.1*sqrt(std::numeric_limits<Scalar>::epsilon());
    
    Scalar sol = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar sols[4];
    const int realRoots = SolveP4(sols, a, b, c, d);
    for(int i=0; i<realRoots; ++i) {
        if(sols[i]>0) {
            sol = sols[i];
        }
    }
    
    //Did we miss a solution, or are we far away from one?
    const auto f = [=] (const Scalar x) {
        return (((x + a)*x + b)*x + c)*x + d;
    };
    const auto df = [=] (const Scalar x) {
        return ((4.*x + 3.*a)*x + 2.*b)*x + c;
    };
    if(!std::isfinite(sol)) {
        sol = (std::max)({std::abs(a), std::abs(b), std::abs(c),
            std::abs(d), tol});
    }
    Scalar fsol;
    int nIter=0;
    for(fsol=f(sol);
        std::abs(fsol)>=tol && std::isfinite(sol) && nIter<100;
        fsol=f(sol)) {
        sol -= fsol / df(sol);
        ++nIter;
    }
    if(sol<=0 || std::abs(fsol)>=tol) {
        sol = std::numeric_limits<Scalar>::quiet_NaN();
    }
    
    return sol;
}


//Explicit template instantiation
template double parametrization::quartic_polynomial<double>(const double, const double, const double, const double);
template float parametrization::quartic_polynomial<float>(const float, const float, const float, const float);
template long double parametrization::quartic_polynomial<long double>(const long double, const long double, const long double, const long double);


#include <math.h>
#include <igl/PI.h>

template <typename Scalar>
static Scalar TwoPi() {
    return 2. * igl::PI;
}
template <typename Scalar>
static Scalar eps() {
    return 50. * std::numeric_limits<Scalar>::epsilon();
}

//-----------------------------------------------------------------------------
// And some additional functions for internal use.
// Your may remove this definitions from here
template <typename Scalar>
static int   SolveP4Bi(Scalar *x, Scalar b, Scalar d);                // solve equation x^4 + b*x^2 + d = 0
template <typename Scalar>
static int   SolveP4De(Scalar *x, Scalar b, Scalar c, Scalar d);    // solve equation x^4 + b*x^2 + c*x + d = 0
template <typename Scalar>
static void  CSqrt( Scalar x, Scalar y, Scalar &a, Scalar &b);        // returns as a+i*s,  sqrt(x+i*y)
template <typename Scalar>
static Scalar N4Step(Scalar x, Scalar a,Scalar b,Scalar c,Scalar d);// one Newton step for x^4 + a*x^3 + b*x^2 + c*x + d
template <typename Scalar>
static Scalar SolveP5_1(Scalar a,Scalar b,Scalar c,Scalar d,Scalar e);    // return real root of x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0


//=============================================================================
// _root3, root3 from http://prografix.narod.ru
//=============================================================================
template <typename Scalar>
static Scalar _root3 ( Scalar x )
{
    Scalar s = 1.;
    while ( x < 1. )
    {
        x *= 8.;
        s *= 0.5;
    }
    while ( x > 8. )
    {
        x *= 0.125;
        s *= 2.;
    }
    Scalar r = 1.5;
    r -= 1./3. * ( r - x / ( r * r ) );
    r -= 1./3. * ( r - x / ( r * r ) );
    r -= 1./3. * ( r - x / ( r * r ) );
    r -= 1./3. * ( r - x / ( r * r ) );
    r -= 1./3. * ( r - x / ( r * r ) );
    r -= 1./3. * ( r - x / ( r * r ) );
    return r * s;
}

template <typename Scalar>
static Scalar root3 ( Scalar x )
{
    if ( x > 0 ) return _root3 ( x ); else
    if ( x < 0 ) return-_root3 (-x ); else
    return 0.;
}


// x - array of size 2
// return 2: 2 real roots x[0], x[1]
// return 0: pair of complex roots: x[0]±i*x[1]
template <typename Scalar>
static int   SolveP2(Scalar *x, Scalar a, Scalar b) {            // solve equation x^2 + a*x + b = 0
    Scalar D = 0.25*a*a - b;
    if (D >= 0) {
        D = sqrt(D);
        x[0] = -0.5*a + D;
        x[1] = -0.5*a - D;
        return 2;
    }
    x[0] = -0.5*a;
    x[1] = sqrt(-D);
    return 0;
}
//---------------------------------------------------------------------------
// x - array of size 3
// In case 3 real roots: => x[0], x[1], x[2], return 3
//         2 real roots: x[0], x[1],          return 2
//         1 real root : x[0], x[1] ± i*x[2], return 1
template <typename Scalar>
static int SolveP3(Scalar *x,Scalar a,Scalar b,Scalar c) {    // solve cubic equation x^3 + a*x^2 + b*x + c = 0
    Scalar a2 = a*a;
    Scalar q  = (a2 - 3*b)/9;
    Scalar r  = (a*(2*a2-9*b) + 27*c)/54;
    // equation x^3 + q*x + r = 0
    Scalar r2 = r*r;
    Scalar q3 = q*q*q;
    Scalar A,B;
    if (r2 <= (q3 + eps<Scalar>())) {//<<-- FIXED!
        Scalar t=r/sqrt(q3);
        if( t<-1) t=-1;
        if( t> 1) t= 1;
        t=acos(t);
        a/=3; q=-2*sqrt(q);
        x[0]=q*cos(t/3)-a;
        x[1]=q*cos((t+TwoPi<Scalar>())/3)-a;
        x[2]=q*cos((t-TwoPi<Scalar>())/3)-a;
        return(3);
    } else {
        //A =-pow(fabs(r)+sqrt(r2-q3),1./3);
        A =-root3(fabs(r)+sqrt(r2-q3));
        if( r<0 ) A=-A;
        B = (A==0? 0 : B=q/A);

        a/=3;
        x[0] =(A+B)-a;
        x[1] =-0.5*(A+B)-a;
        x[2] = 0.5*sqrt(3.)*(A-B);
        if(fabs(x[2])<eps<Scalar>()) { x[2]=x[1]; return(2); }
        return(1);
    }
}// SolveP3(Scalar *x,Scalar a,Scalar b,Scalar c) {
//---------------------------------------------------------------------------
// a>=0!
template <typename Scalar>
static void  CSqrt( Scalar x, Scalar y, Scalar &a, Scalar &b) // returns:  a+i*s = sqrt(x+i*y)
{
    Scalar r  = sqrt(x*x+y*y);
    if( y==0 ) {
        r = sqrt(r);
        if(x>=0) { a=r; b=0; } else { a=0; b=r; }
    } else {        // y != 0
        a = sqrt(0.5*(x+r));
        b = 0.5*y/a;
    }
}
//---------------------------------------------------------------------------
template <typename Scalar>
static int   SolveP4Bi(Scalar *x, Scalar b, Scalar d)    // solve equation x^4 + b*x^2 + d = 0
{
    Scalar D = b*b-4*d;
    if( D>=0 )
    {
        Scalar sD = sqrt(D);
        Scalar x1 = (-b+sD)/2;
        Scalar x2 = (-b-sD)/2;    // x2 <= x1
        if( x2>=0 )                // 0 <= x2 <= x1, 4 real roots
        {
            Scalar sx1 = sqrt(x1);
            Scalar sx2 = sqrt(x2);
            x[0] = -sx1;
            x[1] =  sx1;
            x[2] = -sx2;
            x[3] =  sx2;
            return 4;
        }
        if( x1 < 0 )                // x2 <= x1 < 0, two pair of imaginary roots
        {
            Scalar sx1 = sqrt(-x1);
            Scalar sx2 = sqrt(-x2);
            x[0] =    0;
            x[1] =  sx1;
            x[2] =    0;
            x[3] =  sx2;
            return 0;
        }
        // now x2 < 0 <= x1 , two real roots and one pair of imginary root
            Scalar sx1 = sqrt( x1);
            Scalar sx2 = sqrt(-x2);
            x[0] = -sx1;
            x[1] =  sx1;
            x[2] =    0;
            x[3] =  sx2;
            return 2;
    } else { // if( D < 0 ), two pair of compex roots
        Scalar sD2 = 0.5*sqrt(-D);
        Scalar b2 = 0.5*b;
        CSqrt(-b2, sD2, x[0],x[1]);
        CSqrt(-b2,-sD2, x[2],x[3]);
        return 0;
    } // if( D>=0 )
} // SolveP4Bi(Scalar *x, Scalar b, Scalar d)    // solve equation x^4 + b*x^2 d
//---------------------------------------------------------------------------
#define SWAP(a,b) { t=b; b=a; a=t; }
template <typename Scalar>
static void  dblSort3( Scalar &a, Scalar &b, Scalar &c) // make: a <= b <= c
{
    Scalar t;
    if( a>b ) SWAP(a,b);    // now a<=b
    if( c<b ) {
        SWAP(b,c);            // now a<=b, b<=c
        if( a>b ) SWAP(a,b);// now a<=b
    }
}
//---------------------------------------------------------------------------
template <typename Scalar>
static int   SolveP4De(Scalar *x, Scalar b, Scalar c, Scalar d)    // solve equation x^4 + b*x^2 + c*x + d
{
    //if( c==0 ) return SolveP4Bi(x,b,d); // After that, c!=0
    if( fabs(c) < eps<Scalar>()*(fabs(b)+fabs(d)) ) return SolveP4Bi(x,b,d); // After that, c!=0

    int res3 = SolveP3( x, 2*b, b*b-4*d, -c*c);    // solve resolvent
    // by Viet theorem:  x1*x2*x3=-c*c not equals to 0, so x1!=0, x2!=0, x3!=0
    if( res3>1 )    // 3 real roots,
    {
        dblSort3(x[0], x[1], x[2]);    // sort roots to x[0] <= x[1] <= x[2]
        // Note: x[0]*x[1]*x[2]= c*c > 0
        if( x[0] > 0) // all roots are positive
        {
            Scalar sz1 = sqrt(x[0]);
            Scalar sz2 = sqrt(x[1]);
            Scalar sz3 = sqrt(x[2]);
            // Note: sz1*sz2*sz3= -c (and not equal to 0)
            if( c>0 )
            {
                x[0] = (-sz1 -sz2 -sz3)/2;
                x[1] = (-sz1 +sz2 +sz3)/2;
                x[2] = (+sz1 -sz2 +sz3)/2;
                x[3] = (+sz1 +sz2 -sz3)/2;
                return 4;
            }
            // now: c<0
            x[0] = (-sz1 -sz2 +sz3)/2;
            x[1] = (-sz1 +sz2 -sz3)/2;
            x[2] = (+sz1 -sz2 -sz3)/2;
            x[3] = (+sz1 +sz2 +sz3)/2;
            return 4;
        } // if( x[0] > 0) // all roots are positive
        // now x[0] <= x[1] < 0, x[2] > 0
        // two pair of comlex roots
        Scalar sz1 = sqrt(-x[0]);
        Scalar sz2 = sqrt(-x[1]);
        Scalar sz3 = sqrt( x[2]);

        if( c>0 )    // sign = -1
        {
            x[0] = -sz3/2;
            x[1] = ( sz1 -sz2)/2;        // x[0]±i*x[1]
            x[2] =  sz3/2;
            x[3] = (-sz1 -sz2)/2;        // x[2]±i*x[3]
            return 0;
        }
        // now: c<0 , sign = +1
        x[0] =   sz3/2;
        x[1] = (-sz1 +sz2)/2;
        x[2] =  -sz3/2;
        x[3] = ( sz1 +sz2)/2;
        return 0;
    } // if( res3>1 )    // 3 real roots,
    // now resoventa have 1 real and pair of compex roots
    // x[0] - real root, and x[0]>0,
    // x[1]±i*x[2] - complex roots,
    // x[0] must be >=0. But one times x[0]=~ 1e-17, so:
    if (x[0] < 0) x[0] = 0;
    Scalar sz1 = sqrt(x[0]);
    Scalar szr, szi;
    CSqrt(x[1], x[2], szr, szi);  // (szr+i*szi)^2 = x[1]+i*x[2]
    if( c>0 )    // sign = -1
    {
        x[0] = -sz1/2-szr;            // 1st real root
        x[1] = -sz1/2+szr;            // 2nd real root
        x[2] = sz1/2;
        x[3] = szi;
        return 2;
    }
    // now: c<0 , sign = +1
    x[0] = sz1/2-szr;            // 1st real root
    x[1] = sz1/2+szr;            // 2nd real root
    x[2] = -sz1/2;
    x[3] = szi;
    return 2;
} // SolveP4De(Scalar *x, Scalar b, Scalar c, Scalar d)    // solve equation x^4 + b*x^2 + c*x + d
//-----------------------------------------------------------------------------
template <typename Scalar>
static Scalar N4Step(Scalar x, Scalar a,Scalar b,Scalar c,Scalar d)    // one Newton step for x^4 + a*x^3 + b*x^2 + c*x + d
{
    Scalar fxs= ((4*x+3*a)*x+2*b)*x+c;    // f'(x)
    if (fxs == 0) return x;    //return 1e99; <<-- FIXED!
    Scalar fx = (((x+a)*x+b)*x+c)*x+d;    // f(x)
    return x - fx/fxs;
}
//-----------------------------------------------------------------------------
// x - array of size 4
// return 4: 4 real roots x[0], x[1], x[2], x[3], possible multiple roots
// return 2: 2 real roots x[0], x[1] and complex x[2]±i*x[3],
// return 0: two pair of complex roots: x[0]±i*x[1],  x[2]±i*x[3],
template <typename Scalar>
int   SolveP4(Scalar *x,Scalar a,Scalar b,Scalar c,Scalar d) {    // solve equation x^4 + a*x^3 + b*x^2 + c*x + d by Dekart-Euler method
    // move to a=0:
    Scalar d1 = d + 0.25*a*( 0.25*b*a - 3./64*a*a*a - c);
    Scalar c1 = c + 0.5*a*(0.25*a*a - b);
    Scalar b1 = b - 0.375*a*a;
    int res = SolveP4De( x, b1, c1, d1);
    if( res==4) { x[0]-= a/4; x[1]-= a/4; x[2]-= a/4; x[3]-= a/4; }
    else if (res==2) { x[0]-= a/4; x[1]-= a/4; x[2]-= a/4; }
    else             { x[0]-= a/4; x[2]-= a/4; }
    // one Newton step for each real root:
    if( res>0 )
    {
        x[0] = N4Step(x[0], a,b,c,d);
        x[1] = N4Step(x[1], a,b,c,d);
    }
    if( res>2 )
    {
        x[2] = N4Step(x[2], a,b,c,d);
        x[3] = N4Step(x[3], a,b,c,d);
    }
    return res;
}
//-----------------------------------------------------------------------------
#define F5(t) (((((t+a)*t+b)*t+c)*t+d)*t+e)
//-----------------------------------------------------------------------------
template <typename Scalar>
static Scalar SolveP5_1(Scalar a,Scalar b,Scalar c,Scalar d,Scalar e)    // return real root of x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
{
    int cnt;
    if( fabs(e)<eps<Scalar>() ) return 0;

    Scalar brd =  fabs(a);            // brd - border of real roots
    if( fabs(b)>brd ) brd = fabs(b);
    if( fabs(c)>brd ) brd = fabs(c);
    if( fabs(d)>brd ) brd = fabs(d);
    if( fabs(e)>brd ) brd = fabs(e);
    brd++;                            // brd - border of real roots

    Scalar x0, f0;                    // less than root
    Scalar x1, f1;                    // greater than root
    Scalar x2, f2, f2s;                // next values, f(x2), f'(x2)
    Scalar dx;

    if( e<0 ) { x0 =   0; x1 = brd; f0=e; f1=F5(x1); x2 = 0.01*brd; }    // positive root
    else      { x0 =-brd; x1 =   0; f0=F5(x0); f1=e; x2 =-0.01*brd; }    // negative root

    if( fabs(f0)<eps<Scalar>() ) return x0;
    if( fabs(f1)<eps<Scalar>() ) return x1;

    // now x0<x1, f(x0)<0, f(x1)>0
    // Firstly 10 bisections
    for( cnt=0; cnt<10; cnt++)
    {
        x2 = (x0 + x1) / 2;                    // next point
        //x2 = x0 - f0*(x1 - x0) / (f1 - f0);        // next point
        f2 = F5(x2);                // f(x2)
        if( fabs(f2)<eps<Scalar>() ) return x2;
        if( f2>0 ) { x1=x2; f1=f2; }
        else       { x0=x2; f0=f2; }
    }

    // At each step:
    // x0<x1, f(x0)<0, f(x1)>0.
    // x2 - next value
    // we hope that x0 < x2 < x1, but not necessarily
    do {
        if(cnt++>50) break;
        if( x2<=x0 || x2>= x1 ) x2 = (x0 + x1)/2;    // now  x0 < x2 < x1
        f2 = F5(x2);                                // f(x2)
        if( fabs(f2)<eps<Scalar>() ) return x2;
        if( f2>0 ) { x1=x2; f1=f2; }
        else       { x0=x2; f0=f2; }
        f2s= (((5*x2+4*a)*x2+3*b)*x2+2*c)*x2+d;        // f'(x2)
        if( fabs(f2s)<eps<Scalar>() ) { x2=1e99; continue; }
        dx = f2/f2s;
        x2 -= dx;
    } while(fabs(dx)>eps<Scalar>());
    return x2;
} // SolveP5_1(Scalar a,Scalar b,Scalar c,Scalar d,Scalar e)    // return real root of x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
//-----------------------------------------------------------------------------
template <typename Scalar>
static int   SolveP5(Scalar *x,Scalar a,Scalar b,Scalar c,Scalar d,Scalar e)    // solve equation x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
{
    Scalar r = x[0] = SolveP5_1(a,b,c,d,e);
    Scalar a1 = a+r, b1=b+r*a1, c1=c+r*b1, d1=d+r*c1;
    return 1+SolveP4(x+1, a1,b1,c1,d1);
} // SolveP5(Scalar *x,Scalar a,Scalar b,Scalar c,Scalar d,Scalar e)    // solve equation x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
//-----------------------------------------------------------------------------

