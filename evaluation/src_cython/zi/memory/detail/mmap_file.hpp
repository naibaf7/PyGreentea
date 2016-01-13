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

#ifndef ZI_MEMORY_DETAIL_MMAP_FILE_HPP
#define ZI_MEMORY_DETAIL_MMAP_FILE_HPP 1

#include <zi/memory/detail/file.hpp>

#include <iterator>
#include <cerrno>
#include <cstring>
#include <sys/mman.h>

namespace zi {
namespace mem {
namespace detail {

class mmap_file: public file
{
public:
    inline mmap_file( const std::string& filename, int mode )
        : file( filename, mode )
    {
    }

    bool read( void *buffer,  std::size_t len, std::size_t offset = 0 )
    {
        mutex::guard g( mutex_ );
        int prot = PROT_READ;
        void *m  = mmap( NULL, len, prot, MAP_SHARED, fd_, offset );

        if ( m == MAP_FAILED || m == NULL )
        {
            ZI_THROWF( std::runtime_error,
                       "can't map file: %s ( len: %lld offset: %lld )",
                       filename_.c_str(),
                       len, offset );
        }
        else
        {
            std::memcpy( buffer, m, len );
            if ( munmap( m, len ) < 0 )
            {
                throw std::runtime_error( std::string( "can't munmap file: " ) + filename_ );
            }
        }

        return true;

    }

    template< class OutputIterator >
    bool read_n( OutputIterator first, std::size_t count, std::size_t offset = 0 )
    {

        typedef typename std::iterator_traits< OutputIterator >::value_type value_type;
        std::size_t len = count * sizeof( value_type );

        std::size_t ps = offset % getpagesize();
        if ( ps )
        {
            len += ps;
            offset -= ps;
        }


        mutex::guard g( mutex_ );
        int prot = PROT_READ;
        void *m = mmap( NULL, len, prot, MAP_SHARED, fd_, offset );

        if ( m == MAP_FAILED || m == NULL )
        {
            ZI_THROWF( std::runtime_error,
                       "can't map file: %s ( %s count: %lld offset: %lld ptr: %d )",
                       strerror( errno ),
                       filename_.c_str(),
                       len, offset, fd_ );

        }
        else
        {
            if ( ps )
            {
                char *vc = reinterpret_cast< char* >( m );
                value_type *vm = reinterpret_cast< value_type* >( vc + ps );
                std::copy( vm, vm + count, first );
            }
            else
            {
                value_type *vm = reinterpret_cast< value_type* >( m );
                std::copy( vm, vm + count, first );
            }

            if ( munmap( m, len ) < 0 )
            {
                throw std::runtime_error( std::string( "can't munmap file: " ) + filename_ );
            }
        }
        return true;
    }

    template< class InputIterator >
    bool write( InputIterator first, InputIterator last, std::size_t offset = 0 )
    {
        typedef typename std::iterator_traits< InputIterator >::value_type value_type;

        std::size_t len = static_cast< std::size_t >( last - first + 1 ) * sizeof( value_type );

        mutex::guard g( mutex_ );
        int prot = PROT_WRITE;
        void *m = mmap( NULL, len, prot, MAP_SHARED, fd_, offset );

        if ( m == MAP_FAILED || m == NULL )
        {
            throw std::runtime_error( std::string( "can't mmap file: " ) + filename_ );
        }
        else
        {

            value_type *vm = reinterpret_cast< value_type* >( m );
            std::copy( first, last, vm );

            if ( munmap( m, len ) < 0 )
            {
                throw std::runtime_error( std::string( "can't munmap file: " ) + filename_ );
            }
        }
        return true;
    }

    template< class InputIterator >
    bool write_n( InputIterator first, std::size_t count, std::size_t offset = 0 )
    {
        typedef typename std::iterator_traits< InputIterator >::value_type value_type;
        std::size_t len = count * sizeof( value_type );

        mutex::guard g( mutex_ );
        int prot = PROT_WRITE;
        void *m = mmap( NULL, len, prot, MAP_SHARED, fd_, offset );

        if ( m == MAP_FAILED || m == NULL )
        {
            throw std::runtime_error( std::string( "can't mmap file: " ) + filename_ );
        }
        else
        {
            value_type *vm = reinterpret_cast< value_type* >( m );
            for ( std::size_t i = 0; i < count; ++i )
            {
                vm[ i ] = *( first++ );
            }

            if ( munmap( m, len ) < 0 )
            {
                throw std::runtime_error( std::string( "can't munmap file: " ) + filename_ );
            }
        }
        return true;
    }


    bool write( const void *buffer, std::size_t len, std::size_t offset = 0 )
    {
        mutex::guard g( mutex_ );
        int prot = PROT_WRITE;

        void *m  = mmap( NULL, len, prot, MAP_SHARED, fd_, offset );

        if ( m == MAP_FAILED || m == NULL )
        {
            throw std::runtime_error( std::string( "can't mmap file: " ) + filename_ );
        }
        else
        {
            std::memcpy( m, buffer, len );
            if ( munmap( m, len ) < 0 )
            {
                throw std::runtime_error( std::string( "can't munmap file: " ) + filename_ );
            }
        }

        return true;
    }

    static bool write( const std::string& filename,
                       const void *buffer,
                       const std::size_t len )
    {
        mmap_file f( filename, file::CREAT | file::RDWR | file::TRUNC );
        f.size( len );
        return f.write( buffer, len );
    }

    template< class InputIterator >
    static bool write_n( const std::string& filename,
                         InputIterator first,
                         std::size_t len,
                         std::size_t offset = 0 )
    {
        typedef typename std::iterator_traits< InputIterator >::value_type value_type;
        mmap_file f( filename, file::CREAT | file::RDWR | file::TRUNC );
        f.size( len * sizeof( value_type ) );
        return f.template write_n< InputIterator >( first, len, offset );
    }

    template< class InputIterator >
    static bool write( const std::string& filename,
                       InputIterator first,
                       InputIterator last )
    {
        typedef typename std::iterator_traits< InputIterator >::value_type value_type;

        std::size_t len =
            static_cast< std::size_t >( last - first + 1 ) * sizeof( value_type );

        mmap_file f( filename, file::CREAT | file::RDWR | file::TRUNC );
        f.size( len );
        return f.template write< InputIterator >( first, last );
    }

    static bool read( const std::string& filename, void *buffer,  std::size_t len )
    {
        mmap_file f( filename, file::RDONLY );
        return f.read( buffer, len );
    }

    template< class OutputIterator >
    static bool read_n( const std::string& filename,
                        OutputIterator first,
                        std::size_t count,
                        std::size_t offset = 0 )
    {
        mmap_file f( filename, file::RDONLY );
        return f.template read_n< OutputIterator >( first, count, offset );
    }

    template< class T >
    static bool get( const std::string& filename, T &var, std::size_t offset = 0 )
    {
        mmap_file f( filename, file::RDONLY );
        return f.read( reinterpret_cast< void* >( &var ), sizeof( T ), offset );
    }


};

} // namespace detail

using detail::mmap_file;

} // namespace mem
} // namespace zi

#endif
