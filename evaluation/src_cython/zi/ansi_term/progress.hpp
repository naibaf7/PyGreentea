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

#ifndef ZI_ANSI_TERM_PROGRESS_HPP
#define ZI_ANSI_TERM_PROGRESS_HPP 1

#include <zi/config/config.hpp>
#include <zi/bits/cstdint.hpp>

#include <zi/time/time.hpp>
#include <zi/time/timer.hpp>
#include <zi/ansi_term/utils.hpp>
#include <zi/ansi_term/term_ostream.hpp>
#include <zi/concurrency/mutex.hpp>
#include <zi/concurrency/periodic_function.hpp>
#include <zi/concurrency/thread.hpp>
#include <zi/utility/string_printf.hpp>

namespace zi {
namespace tos {

class progress
{
private:
    const std::string caption_   ;
    int64_t           total_     ;
    int64_t           active_    ;
    int64_t           finished_  ;
    bool              has_active_;
    int64_t           redraw_fq_ ;
    timer::wall       timer_     ;
    double            speed_     ;
    mutex             mutex_     ;
    mutex             draw_mutex_;
    periodic_fn       updater_   ;
public:
    progress( const std::string& caption,
              int64_t total,
              bool has_active = false,
              int64_t redraw_fq = 100 )
        : caption_( caption ),
          total_( total ),
          active_( 0 ),
          finished_( 0 ),
          has_active_( has_active ),
          redraw_fq_( redraw_fq ),
          timer_(),
          speed_( 0 ),
          mutex_(),
          draw_mutex_(),
          updater_( &progress::update, this , redraw_fq )
    {
        zi::thread th( updater_ );
        th.start();
    }

    bool update()
    {
        redraw();
        return true;
    }

    void inc( int64_t finished, int64_t active = 0 )
    {
        mutex::guard g( mutex_ );
        finished_ += finished;
        active_   += active  ;
    }

    void inc_active( int64_t active = 0 )
    {
        mutex::guard g( mutex_ );
        active_   += active  ;
    }

    void redraw()
    {
        mutex::guard dg( draw_mutex_ );

        int width     = detail::get_term_width();
        int bar_width = width - 30;
        bar_width = bar_width > 0 ? bar_width : 0;

        tout << "\r" << string_printf( "%-20s", caption_.c_str() );

        static const char *bar_rotations = "<^__)~....";
        int load_first = ( timer_.elapsed< zi::in_msecs >() / 100 ) % 10;

        if ( bar_width > 0 && total_ > 0 )
        {
            tout << " [";
            int done_width    = 0;
            int started_width = 0;
            {
                mutex::guard g( mutex_ );
                done_width = finished_ * bar_width / total_;
                if ( has_active_ )
                {
                    started_width  = active_ * bar_width / total_;
                    started_width -= done_width;
                }
            }

            int rest_width = bar_width - started_width - done_width;

            if ( done_width > 0 )
            {
                tout << tos::green << std::string( done_width, '#' );
            }

            for ( int i = 0; i < started_width; ++i )
            {
                tout << tos::yellow << bar_rotations[ ( load_first + i ) % 10 ];
            }

            if ( rest_width > 0 )
            {
                tout << tos::red << std::string( rest_width, '.' );
            }

            tout << tos::reset;
        }

        tout << timer_.elapsed< zi::in_secs >();
        tout << "";
        tout << tos::flush;


    }

};

} // namespace tos
} // namespace zi

#endif
