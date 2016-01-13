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

#ifndef ZI_MEMORY_DETAIL_FILE_HPP
#define ZI_MEMORY_DETAIL_FILE_HPP 1

#include <zi/concurrency/mutex.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/utility/exception.hpp>

#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace zi {
namespace mem {
namespace detail {

class file: non_copyable
{
public:
    enum file_mode
    {
        RDONLY = 0x01,
        WRONLY = 0x02,
        RDWR   = 0x04,
        CREAT  = 0x08,
        DIRECT = 0x10,
        TRUNC  = 0x20
    };

protected:
    mutex mutex_;
    int   fd_   ;
    int   mode_ ;
    const std::string filename_;

    std::size_t size_nl()
    {
        struct stat st;
        if ( ::fstat( fd_, &st ) < 0 )
        {
            throw std::runtime_error( std::string( "can't get size of file: " ) + filename_ );
        }
        return st.st_size;
    }


public:
    file( const std::string& filename, int mode )
        : mutex_(),
          fd_( -1 ),
          mode_( mode ),
          filename_( filename )
    {

        int flags = 0;

#define ZI_MEM_DETAIL_ADD_FILE_FLAG( f )     \
        if ( mode & f )                      \
        {                                    \
            flags |= O_##f;                  \
        } static_cast< void >( 0 )           \

        ZI_MEM_DETAIL_ADD_FILE_FLAG( RDONLY );
        ZI_MEM_DETAIL_ADD_FILE_FLAG( WRONLY );
        ZI_MEM_DETAIL_ADD_FILE_FLAG( RDWR   );
        ZI_MEM_DETAIL_ADD_FILE_FLAG( CREAT  );
        ZI_MEM_DETAIL_ADD_FILE_FLAG( TRUNC  );

#undef ZI_MEM_DETAIL_ADD_FILE_FLAG

        if ( mode & DIRECT )
        {
            flags |= O_SYNC | O_RSYNC | O_DSYNC | O_DIRECT;
        }

        const int perms = S_IREAD | S_IWRITE | S_IRGRP | S_IWGRP;

        fd_ = ::open( filename_.c_str(), flags, perms );

        if ( fd_ < 0 )
        {
            ZI_THROWF( std::runtime_error,
                       "can't open file: %s",
                       filename_.c_str() );
        }
    }

    ~file()
    {
        close();
    }

    void close()
    {
        if ( fd_ >= 0 )
        {
            mutex::guard g( mutex_ );

            if ( ::close( fd_ ) < 0 )
            {
                // error
            }
            else
            {
                fd_ = -1;
            }
        }
    }

    void lock()
    {
        mutex::guard g( mutex_ );

        struct flock f;

        f.l_type   = F_RDLCK | F_WRLCK;
        f.l_whence = SEEK_SET;
        f.l_start  = 0;

        f.l_len = 0;

        if ( ::fcntl( fd_, F_SETLK, &f ) < 0 )
        {
            ZI_THROWF( std::runtime_error, "can't lock file: %s", filename_.c_str() );
        }
    }

    std::size_t size()
    {
        mutex::guard g( mutex_ );
        return size_nl();
    }

    void size( std::size_t s )
    {
        mutex::guard g( mutex_ );

        if ( !( mode_ & RDONLY ) )
        {

            if ( ::ftruncate( fd_, s ) < 0 )
            {
                throw std::runtime_error( std::string( "can't set size of file: " ) + filename_ );
            }
        }
    }


    void remove()
    {
        close();
        ::remove( filename_.c_str() );
    }

};


} // namespace detail

using detail::file;

} // namespace mem
} // namespace zi

#endif
