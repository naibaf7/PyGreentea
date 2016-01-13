//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZI_MATH_FFT__HPP
#define ZI_MATH_FFT__HPP 1

#include <zi/math/constants.hpp>
//#include <zi/math/bit_reverse.hpp>
#include <zi/concurrency/mutex.hpp>
#include <zi/bits/type_traits.hpp>
#include <zi/bits/shared_ptr.hpp>
#include <zi/meta/enable_if.hpp>
#include <zi/meta/if.hpp>
#include <zi/meta/bool.hpp>

#include <zi/utility/assert.hpp>
#include <zi/utility/singleton.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/utility/static_assert.hpp>

#include <zi/vl/detail/promote.hpp>

#include <cmath>
#include <vector>
#include <complex>
#include <utility>

namespace zi {
namespace math {

namespace detail {

template< std::size_t Size >
class fft_reverser: non_copyable
{
private:
    std::vector< std::pair< uint32_t, uint32_t > > pairs_   ;
    mutex                                          lock_    ;
    bool                                           computed_;

public:
    void compute()
    {
        mutex::guard g( lock_ );

        if ( computed_ )
        {
            return;
        }

        std::size_t actual_size = ( 1 << Size );

        std::vector< uint32_t > reversers( actual_size );

        for ( std::size_t i = 0; i < actual_size; ++i )
        {
            for ( std::size_t j = 0; j < Size; ++j )
            {
                if ( ( i & ( 1 << j ) ) )
                {
                    reversers[ i ] += ( 1 << ( Size - j - 1 ) );
                }
            }
        }

        for ( std::size_t i = 0; i < actual_size; ++i )
        {
            if ( i < reversers[ i ] )
            {
                pairs_.push_back( std::make_pair( i, reversers[ i ] ) );
            }
        }

        computed_ = true;

    }

    fft_reverser()
        : pairs_(),
          lock_(),
          computed_( false )
    {
    }

    template< class T >
    void operator()( std::vector< T >& data )
    {
        compute();

        FOR_EACH( it, pairs_ )
        {
            std::swap( data[ it->first ], data[ it->second ] );
        }

    }

};

template< std::size_t Size >
class fft_revbin_permute: non_copyable
{
private:
    std::vector< std::pair< uint32_t, uint32_t > > pairs_   ;
    mutex                                          lock_    ;
    bool                                           computed_;

public:
    void compute()
    {
        mutex::guard g( lock_ );

        if ( computed_ )
        {
            return;
        }

        if ( Size > 2 )
        {
            uint32_t n = 1 << Size;
            for ( uint32_t r = 0, x = 1; x < n; ++x )
            {
                for ( uint32_t m = ( n >> 1 ); !( ( r^m ) & m ); m >>= 1 )
                {
                    r ^= m;
                }

                if ( r > x )
                {
                    pairs_.push_back( std::make_pair( x, r ) );
                }
            }
        }

        computed_ = true;

    }

    fft_revbin_permute()
        : pairs_(),
          lock_(),
          computed_( false )
    {
    }

    template< class T >
    void operator()( std::vector< T >& data )
    {
        compute();

        FOR_EACH( it, pairs_ )
        {
            std::swap( data[ it->first ], data[ it->second ] );
        }

    }

};

template< class T, std::size_t S >
class fft_impl: non_copyable
{
private:
    ZI_STATIC_ASSERT( is_floating_point< T >::value, not_floating_point_fft );
    fft_revbin_permute< S >&           permuter_;
    std::vector< std::complex< T > >   roots_   ;

    static const std::size_t           num_elements  = ( 1ull << S );
    static const std::size_t           N             = ( 1ull << S );
    static const std::size_t           half_elements = ( num_elements >> 1 );

public:
    fft_impl()
        : permuter_( singleton< fft_revbin_permute< S > >::instance()  ),
          roots_( num_elements + 1 )
    {

        roots_[ 0 ] = std::complex< T >( 1, 0 );

        for ( std::size_t i = 1; i <= num_elements; ++i )
        {
            roots_[ i ].real() = std::cos( constants< T >::pi() * 2 * i / num_elements );
            roots_[ i ].imag() = std::sin( constants< T >::pi() * 2 * i / num_elements );
        }

    }

    void forward_transform( std::vector< std::complex< T > >& data )
    {
        singleton< fft_reverser< S > >::instance().operator() ( data );
        //permuter_( data );

        std::complex< T > a0;
        std::complex< T > a1;
        std::complex< T > a2;
        std::complex< T > a3;

        std::size_t count  = 0;

        std::size_t ldm = S&1;

        if ( S&1 )
        {
            for ( std::size_t k = 0; k < N; k += 2 )
            {
                a0 = data[ k   ];
                data[ k   ] += data[ k+1 ];
                data[ k+1 ]  = a0 - data[ k+1 ];
            }

            if ( S >= 3 )
            {
                for ( std::size_t r = 0; r < N; r += 8 )
                {
                    a0 = data[ r   ];
                    a1 = data[ r+4 ];
                    a2 = data[ r+2 ];
                    a3 = data[ r+6 ];

                    data[ r   ] = (a0+a2) + (a1+a3);
                    data[ r+4 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() = -a1.imag();
                    a3.imag() =  a1.real();

                    data[ r+2 ] = (a0-a2) + a3;
                    data[ r+6 ] = (a0-a2) - a3;

                    a0 = data[ r+1 ];

                    T sin_cos = roots_[ N >> 3 ].real();

                    a1 = data[ r+5 ];
                    a1.real() -= a1.imag();
                    a1.imag() += data[ r+5 ].real();
                    a1 *= sin_cos;

                    a2.real() = -data[ r+3 ].imag();
                    a2.imag() =  data[ r+3 ].real();

                    a3 = -data[ r+7 ];
                    a3.real() += a3.imag();
                    a3.imag() += data[ r+7 ].real();

                    count += 4;

                    data[ r+1 ] = (a0+a2) + (a1+a3);
                    data[ r+5 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() = -a1.imag();
                    a3.imag() =  a1.real();

                    data[ r+3 ] = (a0-a2) + a3;
                    data[ r+7 ] = (a0-a2) - a3;
                }
                ldm = 3;
            }
        }
        else
        {
            for ( std::size_t r = 0; r < N; r += 4 )
            {
                a0 = data[ r   ];
                a1 = data[ r+2 ];
                a2 = data[ r+1 ];
                a3 = data[ r+3 ];

                data[ r   ] = (a0+a2) + (a1+a3);
                data[ r+2 ] = (a0+a2) - (a1+a3);

                a1 -= a3;
                a3.real() = -a1.imag();
                a3.imag() =  a1.real();

                data[ r+1 ] = (a0-a2) + a3;
                data[ r+3 ] = (a0-a2) - a3;
            }
            ldm = 2;
        }

        ldm += 2;

        for ( ; ldm <= S; ldm += 2 )
        {
            std::size_t m  = 1 << ldm;
            std::size_t m4 = m >> 2;
            std::size_t da = 1 << ( S - ldm );
            std::size_t an = da;

            for ( std::size_t r = 0; r < N; r += m )
            {
                const std::size_t i0 = r;
                const std::size_t i1 = i0 + m4;
                const std::size_t i2 = i1 + m4;
                const std::size_t i3 = i2 + m4;

                a0 = data[ i0 ];
                a1 = data[ i2 ];
                a2 = data[ i1 ];
                a3 = data[ i3 ];

                data[ i0 ] = (a0+a2) + (a1+a3);
                data[ i2 ] = (a0+a2) - (a1+a3);

                a1 -= a3;
                a3.real() = -a1.imag();
                a3.imag() =  a1.real();

                data[ i1 ] = (a0-a2) + a3;
                data[ i3 ] = (a0-a2) - a3;
            }

            for ( std::size_t j = 1; j < m4; ++j, an += da )
            {
                const std::complex< T >& e  = roots_[ an       ];
                const std::complex< T >& e2 = roots_[ an+da    ];
                const std::complex< T >& e3 = roots_[ an+da+da ];

                for ( std::size_t r = 0, i0 = j+r; r < N; r += m, i0 += m )
                {
                    const std::size_t i1 = i0 + m4;
                    const std::size_t i2 = i1 + m4;
                    const std::size_t i3 = i2 + m4;

                    a0 = data[ i0 ];
                    a1 = data[ i2 ];
                    a2 = data[ i1 ];
                    a3 = data[ i3 ];

                    a1 *= e;
                    a2 *= e2;
                    a3 *= e3;

                    count += 12;

                    data[ i0 ] = (a0+a2) + (a1+a3);
                    data[ i2 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() = -a1.imag();
                    a3.imag() =  a1.real();

                    data[ i1 ] = (a0-a2) + a3;
                    data[ i3 ] = (a0-a2) - a3;
                }
            }
        }

        std::cout << "Total Mult: " << count << "\n";
    }

    void inverse_transform( std::vector< std::complex< T > >& data )
    {
        permuter_( data );

        std::size_t count = 0;
        std::size_t ldm = S&1;

        if ( ldm != 0 )
        {
            for ( std::size_t k = 0; k < N; ++k )
            {
                std::complex< T > t( data[ k + 1 ] );
                data[ k ]     += data[ k + 1 ];
                data[ k + 1 ]  = t - data[ k + 1 ];
            }
        }

        ldm += 2;
        for ( ; ldm <= S; ldm += 2 )
        {
            std::size_t m  = 1 << ldm;
            std::size_t m4 = m >> 2;

            for ( std::size_t j = 0; j < m4; ++j )
            {
                const std::complex< T >& e  = roots_[ j ];
                const std::complex< T >& e2 = roots_[ 2*j ];
                const std::complex< T >& e3 = roots_[ 3*j ];

                for ( std::size_t r = 0, i0 = j+r; r < N; r += m, i0 += m )
                {
                    const std::size_t i1 = i0 + m4;
                    const std::size_t i2 = i1 + m4;
                    const std::size_t i3 = i2 + m4;

                    std::complex< T > a0( data[ i0 ] );
                    std::complex< T > a1( data[ i2 ] );
                    std::complex< T > a2( data[ i1 ] );
                    std::complex< T > a3( data[ i3 ] );

                    a1 *= e;
                    a2 *= e2;
                    a3 *= e3;

                    count += 12;

                    data[ i0 ] = (a0+a2) + (a1+a3);
                    data[ i2 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() =  a1.imag();
                    a3.imag() = -a1.real();

                    data[ i1 ] = (a0-a2) + a3;
                    data[ i3 ] = (a0-a2) - a3;
                }
            }
        }

        std::cout << "Total Mult: " << count << "\n";

    }

};

template< class T, std::size_t S >
class fft_radix_4_impl: non_copyable
{
private:
    ZI_STATIC_ASSERT( is_floating_point< T >::value, not_floating_point_fft );
    fft_revbin_permute< S >&           permuter_;
    std::vector< std::complex< T > >   roots_   ;

    static const std::size_t           num_elements  = ( 1ull << S );
    static const std::size_t           N             = ( 1ull << S );
    static const std::size_t           half_elements = ( num_elements >> 1 );

public:
    fft_radix_4_impl()
        : permuter_( singleton< fft_revbin_permute< S > >::instance()  ),
          roots_( num_elements + 1 )
    {

        roots_[ 0 ] = std::complex< T >( 1, 0 );

        for ( std::size_t i = 1; i <= num_elements; ++i )
        {
            roots_[ i ].real() = std::cos( constants< T >::pi() * 2 * i / num_elements );
            roots_[ i ].imag() = std::sin( constants< T >::pi() * 2 * i / num_elements );
        }

    }

    void forward_transform( std::vector< std::complex< T > >& data )
    {
        singleton< fft_reverser< S > >::instance().operator() ( data );
        //permuter_( data );

        std::complex< T > a0;
        std::complex< T > a1;
        std::complex< T > a2;
        std::complex< T > a3;

        std::size_t count  = 0;

        std::size_t ldm = S&1;

        if ( S&1 )
        {
            for ( std::size_t k = 0; k < N; k += 2 )
            {
                a0 = data[ k   ];
                data[ k   ] += data[ k+1 ];
                data[ k+1 ]  = a0 - data[ k+1 ];
            }
        }

        ldm += 2;

        for ( ; ldm <= S; ldm += 2 )
        {
            std::size_t m  = 1 << ldm;
            std::size_t m4 = m >> 2;
            std::size_t da = 1 << ( S - ldm );
            std::size_t an = da;

            for ( std::size_t j = 0; j < m4; ++j, an += da )
            {
                const std::complex< T >& e  = roots_[ an       ];
                const std::complex< T >& e2 = roots_[ an+da    ];
                const std::complex< T >& e3 = roots_[ an+da+da ];

                for ( std::size_t r = 0, i0 = j+r; r < N; r += m, i0 += m )
                {
                    const std::size_t i1 = i0 + m4;
                    const std::size_t i2 = i1 + m4;
                    const std::size_t i3 = i2 + m4;

                    a0 = data[ i0 ];
                    a1 = data[ i2 ];
                    a2 = data[ i1 ];
                    a3 = data[ i3 ];

                    a1 *= e;
                    a2 *= e2;
                    a3 *= e3;

                    count += 12;

                    data[ i0 ] = (a0+a2) + (a1+a3);
                    data[ i2 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() = -a1.imag();
                    a3.imag() =  a1.real();

                    data[ i1 ] = (a0-a2) + a3;
                    data[ i3 ] = (a0-a2) - a3;
                }
            }
        }

        std::cout << "Total Mult: " << count << "\n";
    }

    void inverse_transform( std::vector< std::complex< T > >& data )
    {
        permuter_( data );

        std::size_t count = 0;
        std::size_t ldm = S&1;

        if ( ldm != 0 )
        {
            for ( std::size_t k = 0; k < N; ++k )
            {
                std::complex< T > t( data[ k + 1 ] );
                data[ k ]     += data[ k + 1 ];
                data[ k + 1 ]  = t - data[ k + 1 ];
            }
        }

        ldm += 2;
        for ( ; ldm <= S; ldm += 2 )
        {
            std::size_t m  = 1 << ldm;
            std::size_t m4 = m >> 2;

            for ( std::size_t j = 0; j < m4; ++j )
            {
                const std::complex< T >& e  = roots_[ j ];
                const std::complex< T >& e2 = roots_[ 2*j ];
                const std::complex< T >& e3 = roots_[ 3*j ];

                for ( std::size_t r = 0, i0 = j+r; r < N; r += m, i0 += m )
                {
                    const std::size_t i1 = i0 + m4;
                    const std::size_t i2 = i1 + m4;
                    const std::size_t i3 = i2 + m4;

                    std::complex< T > a0( data[ i0 ] );
                    std::complex< T > a1( data[ i2 ] );
                    std::complex< T > a2( data[ i1 ] );
                    std::complex< T > a3( data[ i3 ] );

                    a1 *= e;
                    a2 *= e2;
                    a3 *= e3;

                    count += 12;

                    data[ i0 ] = (a0+a2) + (a1+a3);
                    data[ i2 ] = (a0+a2) - (a1+a3);

                    a1 -= a3;
                    a3.real() =  a1.imag();
                    a3.imag() = -a1.real();

                    data[ i1 ] = (a0-a2) + a3;
                    data[ i3 ] = (a0-a2) - a3;
                }
            }
        }

        std::cout << "Total Mult: " << count << "\n";

    }

};

template< class T, std::size_t S >
class fft_split_radix_impl: non_copyable
{
private:
    ZI_STATIC_ASSERT( is_floating_point< T >::value, not_floating_point_fft     );
    fft_reverser< S >&                 reverser_;
    std::vector< std::complex< T > >   roots_   ;

    static const std::size_t           N                = ( 1ull << S );
    static const std::size_t           num_elements     = ( 1ull << S );
    static const std::size_t           half_elements    = ( num_elements >> 1 );
    static const std::size_t           quarter_elements = ( num_elements >> 2 );

public:
    fft_split_radix_impl()
        : reverser_( singleton< fft_reverser< S > >::instance()  ),
          roots_( num_elements + 1 )
    {

        roots_[ 0 ] = std::complex< T >( 1, 0 );

        std::complex< T > mult( std::cos( constants< T >::pi() * 2 / num_elements ),
                                std::sin( constants< T >::pi() * 2 / num_elements ) );

        for ( std::size_t i = 1; i <= half_elements; ++i )
        {
            roots_[ i ] = roots_[ i - 1 ] * mult;
        }

        for ( std::size_t i = 0; i <= half_elements; ++i )
        {
            roots_[ i + half_elements ] = -roots_[ i ];
        }
    }

    void forward_transform( std::vector< std::complex< T > >& data )
    {

        reverser_( data );

        std::size_t count  = 0;

        for ( std::size_t i = 1; i <= S; ++i )
        {
            std::size_t m   = ( 1 << i );
            std::size_t md2 = m / 2;

            std::size_t increment = ( 1 << ( S - i ) );

            for ( std::size_t k = 0; k < num_elements; k += m )
            {
                std::size_t index = 0;
                for ( std::size_t j = k; j < md2 + k; ++j, index += increment)
                {
                    std::complex< T > t( roots_[ index ] * data[ j + md2 ] );
                    data[ j + md2 ] = data[ j ] - t;
                    data[ j ] += t;
                    count += 4;
                }
            }
        }

        std::cout << "Total Mult: " << count << "\n";
    }

    void inverse_transform( std::vector< std::complex< T > >& data )
    {
        reverser_( data );

        for ( std::size_t i = 1; i <= S; ++i )
        {
            std::size_t m   = ( 1 << i );
            std::size_t md2 = m / 2;

            std::size_t increment = ( 1 << ( S - i ) );

            for ( std::size_t k = 0; k < num_elements; k += m )
            {
                std::size_t index = num_elements;
                for ( std::size_t j = k; j < md2 + k; ++j, index -= increment)
                {
                    std::complex< T > t( roots_[ index ] * data[ j + md2 ] );
                    data[ j + md2 ] = data[ j ] - t;
                    data[ j ] += t;
                }
            }
        }

        T factor = static_cast< T >( 1 ) / num_elements;
        FOR_EACH( it, data )
        {
            (*it) *= factor;
        }

    }


};

template< class T, std::size_t S >
class fft_transformer;


template< class T >
class fft_transformer< T, 26 >: non_copyable
{
public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_impl< T, 26 > >::instance().forward_transform( data );
        data.resize( len );
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_impl< T, 26 > >::instance().inverse_transform( data );
        data.resize( len );
    }
};

template< class T, std::size_t S >
class fft_transformer: non_copyable
{
private:
    ZI_STATIC_ASSERT( S < 26, too_large_fft_transformer );

public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_impl< T, S > >::instance().forward_transform( data );
        }
        else
        {
            fft_transformer< T, S+1 >::forward( data );
        }
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_impl< T, S > >::instance().inverse_transform( data );
        }
        else
        {
            fft_transformer< T, S+1 >::inverse( data );
        }
    }
};

template< class T, std::size_t S >
class fft_split_radix_transformer;


template< class T >
class fft_split_radix_transformer< T, 26 >: non_copyable
{
public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_split_radix_impl< T, 26 > >::instance().forward_transform( data );
        data.resize( len );
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_split_radix_impl< T, 26 > >::instance().inverse_transform( data );
        data.resize( len );
    }
};

template< class T, std::size_t S >
class fft_split_radix_transformer: non_copyable
{
private:
    ZI_STATIC_ASSERT( S < 26, too_large_fft_split_radix_transformer );

public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_split_radix_impl< T, S > >::instance().forward_transform( data );
        }
        else
        {
            fft_split_radix_transformer< T, S+1 >::forward( data );
        }
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_split_radix_impl< T, S > >::instance().inverse_transform( data );
        }
        else
        {
            fft_split_radix_transformer< T, S+1 >::inverse( data );
        }
    }
};

template< class T, std::size_t S >
class fft_radix_4_transformer;


template< class T >
class fft_radix_4_transformer< T, 26 >: non_copyable
{
public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_radix_4_impl< T, 26 > >::instance().forward_transform( data );
        data.resize( len );
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        std::size_t len = data.size();
        data.resize( 1 << 26 );
        singleton< fft_radix_4_impl< T, 26 > >::instance().inverse_transform( data );
        data.resize( len );
    }
};

template< class T, std::size_t S >
class fft_radix_4_transformer: non_copyable
{
private:
    ZI_STATIC_ASSERT( S < 26, too_large_fft_radix_4_transformer );

public:
    static inline void forward( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_radix_4_impl< T, S > >::instance().forward_transform( data );
        }
        else
        {
            fft_radix_4_transformer< T, S+1 >::forward( data );
        }
    }

    static inline void inverse( std::vector< std::complex< T > >& data )
    {
        if ( data.size() <= ( 1 << S ) )
        {
            data.resize( 1 << S );
            singleton< fft_radix_4_impl< T, S > >::instance().inverse_transform( data );
        }
        else
        {
            fft_radix_4_transformer< T, S+1 >::inverse( data );
        }
    }
};

} // namespace detail

template< class T >
void fft_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_transformer< T, 0 >::forward( data );
}

template< class T >
void ifft_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_transformer< T, 0 >::inverse( data );
}

template< class T >
void fft_inverse_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_transformer< T, 0 >::inverse( data );
}

template< class T >
void fft_split_radix_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_split_radix_transformer< T, 0 >::forward( data );
}

template< class T >
void ifft_split_radix_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_split_radix_transformer< T, 0 >::inverse( data );
}

template< class T >
void fft_inverse_split_radix_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_split_radix_transformer< T, 0 >::inverse( data );
}

template< class T >
void fft_radix_4_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_radix_4_transformer< T, 0 >::forward( data );
}

template< class T >
void ifft_radix_4_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_radix_4_transformer< T, 0 >::inverse( data );
}

template< class T >
void fft_inverse_radix_4_transform( std::vector< std::complex< T > >& data )
{
    detail::fft_radix_4_transformer< T, 0 >::inverse( data );
}

namespace detail {

template< class T1, class T2 >
class conv_result_reference
{
private:
    ZI_STATIC_ASSERT( is_scalar< T1 >::value, not_scalar_type  );
    ZI_STATIC_ASSERT( is_scalar< T2 >::value, not_scalar_type  );

    typedef std::complex< typename vl::detail::promote< T1, T2 >::type > promoted_complex   ;
    shared_ptr< std::vector< promoted_complex > >                        complex_result_ptr_;

public:
    explicit conv_result_reference( const std::vector< T1 >& a, const std::vector< T2 >& b )
        : complex_result_ptr_( new std::vector< promoted_complex > )
    {
        std::size_t alen = a.size();
        std::size_t blen = b.size();

        if ( alen > 0 && blen > 0 )
        {
            std::size_t len = alen + blen - 1;
            std::vector< promoted_complex >  ca( len );
            std::vector< promoted_complex >& cb = *complex_result_ptr_;

            for ( std::size_t i = 0; i < alen; ++i )
            {
                ca[ i ] = a[ i ];
            }

            fft_transform( ca );

            cb.resize( len );

            for ( std::size_t i = 0; i < blen; ++i )
            {
                cb[ i ] = b[ i ];
            }

            fft_transform( cb );

            for ( std::size_t i = 0; i < cb.size(); ++i )
            {
                cb[ i ] *= ca[ i ];
            }

            fft_inverse_transform( cb );

            cb.resize( len );
        }
    }

    template< class T >
    void fill_vector( std::vector< T >& out,
                      typename meta::enable_if< is_integral< T > >::type* = 0 ) const
    {
        const std::vector< promoted_complex >& cb = *complex_result_ptr_;
        out.resize( cb.size() );

        for ( std::size_t i = 0; i < cb.size(); ++i )
        {
            out[ i ] = static_cast< T >( cb[ i ].real() + 0.5 );
        }
    }

    template< class T >
    void fill_vector( std::vector< T >& out,
                      typename meta::enable_if< is_floating_point< T > >::type* = 0 ) const
    {
        const std::vector< promoted_complex >& cb = *complex_result_ptr_;
        out.resize( cb.size() );

        for ( std::size_t i = 0; i < cb.size(); ++i )
        {
            out[ i ] = static_cast< T >( cb[ i ].real() );
        }
    }

    template< class T >
    operator std::vector< T >() const
    {
        std::vector< T > result;
        this->template fill_vector< T >( result );
        return result;
    }

};

} // namespace detail

template< class T1, class T2 >
inline detail::conv_result_reference< T1, T2 >
conv( const std::vector< T1 >& a, const std::vector< T2 >& b )
{
    return detail::conv_result_reference< T1, T2 >( a, b );
}


} // namespace math
} // namespace zi

#endif
