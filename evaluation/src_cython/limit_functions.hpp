#pragma once

#include <cstddef>

struct linear
{
private:
    double coef_;

public:
    linear( double f )
        : coef_(f)
    {}

    std::size_t operator()( float v ) const
    {
        if ( v < 0.3 ) return 0;
        return v * coef_;
    }
};

struct square
{
private:
    double coef_;

public:
    square( double f )
        : coef_(f)
    {}

    std::size_t operator()( float v ) const
    {
        if ( v < 0.3 ) return 0;
        return v * v * coef_;
    }
};


struct power
{
private:
    float pow_  ;
    float max_;

public:
    power( float f, float t )
        : pow_(f)
        , max_(t)
    {}

    std::size_t operator()( float v ) const
    {
        if ( v < 0.3 ) return 0;
        //v; //= (v-0.7-0.01) / 0.69;

        return v * max_;
    }
};


struct exponent
{
private:
    float       rate_ ;
    std::size_t thold_;

public:
    exponent( float f, std::size_t t )
        : rate_(f)
        , thold_(t)
    {}

    std::size_t operator()( float v ) const
    {
        if ( v < 0.3 ) return 0;

        float x = v - 0.3;

        float max = thold_;
        float rate = rate_;

        return std::exp(rate * x) + 100;
    }
};


struct const_above_threshold
{
private:
    float       min_  ;
    std::size_t thold_;

public:
    const_above_threshold( float f, std::size_t t )
        : min_(f)
        , thold_(t)
    {}

    std::size_t operator()( float v ) const
    {
        return v > min_ ? thold_ : 0;
    }
};

struct square_fn
{
private:
    float min_;
    float max_;
    float off_;

public:
    square_fn( float f, std::size_t m, float o)
        : min_(f), max_(m), off_(o)
    {}

    float operator()( float v ) const
    {
        if ( v < min_ ) return 0;

        float d = 1;
        d -= min_;

        return off_ + max_ * (v-min_)*(v-min_) / d / d;
    }
};



std::size_t custom_fn( float v )
{
    if ( v < 0.3 ) return 0;

    if ( v > 0.98 ) return 5000;
    if ( v > 0.97 ) return 2000;
    if ( v > 0.96 ) return 1500;
    if ( v > 0.95 ) return 500;

    if ( v > 0.5 ) return 250;
    return 100;
}

std::size_t custom2_fn( float v )
{
    if ( v < 0.3 ) return 0;

    float x = v - 1;

    float max = 50000;
    float rate = 5;

    return std::exp(rate * x) * (max - 250) + 250;
}


std::size_t limit_fn( float v )
{
    //return 50;

    // size threshold based on affinity
    if ( v > 1 )
    {
        return 2000;
    }

    if ( v < 0.3 )
    {
        return 0;
    }

    if ( v < 0.5 )
    {
        return 50;
    }

    v *= 10;


    return static_cast<std::size_t>(50+v*v*v);
}


std::size_t limit_fn2( float v )
{
    if ( v < 0.3 ) return 0;

    if ( v > 0.98 ) return 5000;
    if ( v > 0.97 ) return 2000;
    if ( v > 0.96 ) return 1500;
    if ( v > 0.95 ) return 500;

    if ( v > 0.5 ) return 250;
    //return 500;
    return 100;

    // size threshold based on affinity
    if ( v > 1 )
    {
        return 2000;
    }

    if ( v < 0.3 )
    {
        return 0;
    }

    if ( v < 0.5 )
    {
        return 150;
    }

    v *= 10;


    return static_cast<std::size_t>(50+v*v*v);
}


std::size_t limit_fn4( float v )
{
    if ( v < 0.3 ) return 0;

    return 250 + 50000.0 * (v-0.3)*(v-0.3) / 0.7 / 0.7;


    if ( v > 0.98 ) return 5000;
    if ( v > 0.97 ) return 2000;
    if ( v > 0.96 ) return 1500;
    if ( v > 0.95 ) return 500;

    if ( v > 0.5 ) return 250;
    //return 500;
    return 100;

    // size threshold based on affinity
    if ( v > 1 )
    {
        return 2000;
    }

    if ( v < 0.3 )
    {
        return 0;
    }

    if ( v < 0.5 )
    {
        return 150;
    }

    v *= 10;


    return static_cast<std::size_t>(50+v*v*v);
}



std::size_t limit_fn3( float v )
{
    if ( v < 0.3 ) return 0;
    return 100;

    if ( v > 0.98 ) return 5000;
    if ( v > 0.97 ) return 2000;
    if ( v > 0.96 ) return 1500;
    if ( v > 0.95 ) return 500;

    if ( v > 0.5 ) return 250;
    //return 500;
    return 100;

    // size threshold based on affinity
    if ( v > 1 )
    {
        return 2000;
    }

    if ( v < 0.3 )
    {
        return 0;
    }

    if ( v < 0.5 )
    {
        return 150;
    }

    v *= 10;


    return static_cast<std::size_t>(50+v*v*v);
}
