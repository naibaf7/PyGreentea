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

#ifndef ZI_WEB_SERVER_POLL_HPP
#define ZI_WEB_SERVER_POLL_HPP 1

#include <zi/web/server/aio.hpp>
#include <zi/concurrency/spinlock.hpp>
#include <zi/concurrency/mutex.hpp>
#include <zi/concurrency/thread.hpp>
#include <zi/concurrency/condition_variable.hpp>
#include <zi/concurrency/runnable.hpp>
#include <zi/utility/assert.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/bits/unordered_map.hpp>
#include <zi/bits/function.hpp>
#include <zi/bits/bind.hpp>
#include <zi/bits/ref.hpp>

#include <list>
#include <utility>

namespace zi {
namespace web {
namespace server {
namespace poll_ {

class poll_manager: public runnable
{
private:

    struct poll_tag;

    enum status
    {
        IDLE     = 0,
        RUNNING  = 1,
        STOPPING = 2,
        STOPPED  = 3
    };

    aio                aio_           ;
    int                pending_remove_;
    status             status_        ;
    spinlock           mutex_         ;
    mutex              status_mutex_  ;
    condition_variable status_cv_     ;

    unordered_map< int, function< void() > > read_callbacks_;
    unordered_map< int, function< void() > > write_callbacks_;

public:

    poll_manager():
        aio_(),
        pending_remove_( -1 ),
        status_( IDLE ),
        mutex_(),
        status_mutex_(),
        status_cv_(),
        read_callbacks_(),
        write_callbacks_()
    {
    }

    ~poll_manager()
    {
        stop();
    }

    void register_read_callback( int fd, const function< void() > &fn )
    {
        spinlock::guard g( mutex_ );

        aio_.add< aio::RDONLY >( fd );

        ZI_VERIFY ( read_callbacks_.count( fd ) == 0 );
        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            read_callbacks_.insert( std::make_pair( fd, fn ) );
        }
    }

    void register_read_callback( int fd, const reference_wrapper< function< void() > > &fn )
    {
        register_read_callback( fd, fn.get() );
    }

    void register_write_callback( int fd, const function< void() > &fn )
    {
        spinlock::guard g( mutex_ );

        aio_.add< aio::WRONLY >( fd );

        ZI_VERIFY ( write_callbacks_.count( fd ) == 0 );
        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            write_callbacks_.insert( std::make_pair( fd, fn ) );
        }
    }

    void register_write_callback( int fd, const reference_wrapper< function< void() > > &fn )
    {
        register_write_callback( fd, fn.get() );
    }

    void register_rdwr_callback( int fd,
                                 const function< void() > &rdfn,
                                 const function< void() > &wrfn )
    {
        spinlock::guard g( mutex_ );

        aio_.add< aio::RDWR >( fd );

        ZI_VERIFY ( read_callbacks_.count( fd )  == 0 );
        ZI_VERIFY ( write_callbacks_.count( fd ) == 0 );

        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            read_callbacks_.insert ( std::make_pair( fd, rdfn ) );
            write_callbacks_.insert( std::make_pair( fd, wrfn ) );
        }
    }

    void register_rdwr_callback( int fd,
                                 const reference_wrapper< function< void() > > &rdfn,
                                 const reference_wrapper< function< void() > > &wrfn )
    {
        register_rdwr_callback( fd, rdfn.get(), wrfn.get() );
    }


    void register_callbacks( int fd,
                             const function< void() > &rdfn,
                             const function< void() > &wrfn )
    {
        register_rdwr_callback( fd, rdfn, wrfn );
    }

    void register_callbacks( int fd,
                             const reference_wrapper< function< void() > > &rdfn,
                             const reference_wrapper< function< void() > > &wrfn )
    {
        register_rdwr_callback( fd, rdfn.get(), wrfn.get() );
    }


    void unregister_read_callback( int fd )
    {
        spinlock::guard g( mutex_ );

        ZI_VERIFY( aio_.remove< aio::RDONLY >( fd ) );
        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            read_callbacks_.erase( fd );
        }
    }


    void unregister_write_callback( int fd )
    {
        spinlock::guard g( mutex_ );

        ZI_VERIFY( aio_.remove< aio::WRONLY >( fd ) );
        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            write_callbacks_.erase( fd );
        }
    }

    void unregister_rdwr_callback( int fd )
    {
        spinlock::guard g( mutex_ );

        ZI_VERIFY( aio_.remove< aio::RDWR >( fd ) );
        {
            spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( fd ) );
            read_callbacks_.erase( fd );
            write_callbacks_.erase( fd );
        }
    }

    void remove_fd( int fd )
    {
        unregister_rdwr_callback( fd );
    }

    void stop()
    {
        mutex::guard g( status_mutex_ );

        if ( status_ == IDLE || status_ == STOPPED )
        {
            status_ = STOPPED;
            return;
        }

        if ( status_ == RUNNING )
        {
            status_ = STOPPING;
            aio_.interrupt();
        }

        while ( status_ != STOPPED )
        {
            status_cv_.wait( g );
        }

    }

    void run()
    {

        {
            mutex::guard g( status_mutex_ );
            ZI_VERIFY( status_ == IDLE );
            status_ = RUNNING;
        }

        std::list< int > readable, writable;

        while ( 1 )
        {

            {
                mutex::guard g( status_mutex_ );

                ZI_VERIFY( status_ != IDLE && status_ != STOPPED );

                if ( status_ == STOPPING )
                {
                    status_ = STOPPED;
                    status_cv_.notify_all();
                    break;
                }
            }

            readable.clear();
            writable.clear();

            if ( aio_.wait( readable, writable ) )
            {

                FOR_EACH( it, readable )
                {
                    spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( *it ) );
                    if ( read_callbacks_.count( *it ) )
                    {
                        (read_callbacks_[ *it ])();
                    }
                }

                FOR_EACH( it, writable )
                {
                    spinlock::pool< poll_tag >::guard g( static_cast< std::size_t >( *it ) );
                    if ( write_callbacks_.count( *it ) )
                    {
                        (write_callbacks_[ *it ])();
                    }
                }
            }

        }


    }

};

namespace {
static shared_ptr< poll_manager > poll_ptr( new poll_manager );
}

} // namespace poll_

namespace {
static poll_::poll_manager& poll = *poll_::poll_ptr;
static ::zi::thread poll_thread( poll_::poll_ptr );
}

} // namespace server
} // namespace web
} // namespace zi

#endif

