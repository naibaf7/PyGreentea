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

#ifndef ZI_WEB_SERVER_SERVER_SOCKET_HPP
#define ZI_WEB_SERVER_SERVER_SOCKET_HPP 1

namespace zi {
namespace web {

class server_socket
{
private:
    int port_;
    int server_socket_;
    int send_timeout_;
    int recv_timeout_;
    int retry_limit_;
    int retry_delay_;
    int send_buffer_;
    int recv_buffer_;
    int sock_pair_[2];

    void close()
    {
    }

public:

    server_socket( int port, int send_to = 0, int recv_to = 0, int retry_limit = 0, int retry_delay = 0 ):
        port_( port ),
        server_socket_( -1 ),
        send_timeout_( send_to ),
        recv_timeout_( recv_to ),
        retry_limit_( retry_limit ),
        retry_delay_( retry_delay ),
        send_buffer_( 0 ),
        recv_buffer_( 0 ),
    {
        sock_pair_[0] = sock_pair_[1] = -1;
    }

    ~server_socket()
    {
        close();
    }


    bool is_open() const
    {
        return false;
    }

    bool peak()
    {
        return is_open();
    }
};

} // namespace web
} // namespace zi

#endif

