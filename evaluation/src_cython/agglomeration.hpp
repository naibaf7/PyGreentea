#pragma once

#include "types.hpp"

#include <zi/disjoint_sets/disjoint_sets.hpp>
#include <map>
#include <vector>
#include <set>

template< typename ID, typename F, typename L, typename M >
inline void merge_segments( const volume_ptr<ID>& seg_ptr,
                            const region_graph_ptr<ID,F> rg_ptr,
                            std::vector<std::size_t>& counts,
                            const L& tholds,
                            const M& lowt )
{
    zi::disjoint_sets<ID> sets(counts.size());

    typename region_graph<ID,F>::iterator rit = rg_ptr->begin();

    region_graph<ID,F>& rg  = *rg_ptr;

    for ( auto& it: tholds )
    {
        std::size_t size = static_cast<std::size_t>(it.first);
        F           thld = static_cast<F>(it.second);

        while ( (rit != rg.end()) && ( std::get<0>(*rit) > thld) )
        {
            ID s1 = sets.find_set(std::get<1>(*rit));
            ID s2 = sets.find_set(std::get<2>(*rit));

            if ( s1 != s2 && s1 && s2 )
            {
                if ( (counts[s1] < size) || (counts[s2] < size) )
                {
                    counts[s1] += counts[s2];
                    counts[s2]  = 0;
                    ID s = sets.join(s1,s2);
                    std::swap(counts[s], counts[s1]);
                }
            }
            ++rit;
        }
    }

    std::cout << "Done with merging" << std::endl;

    std::vector<ID> remaps(counts.size());

    ID next_id = 1;

    std::size_t low = static_cast<std::size_t>(lowt);

    for ( ID id = 0; id < counts.size(); ++id )
    {
        ID s = sets.find_set(id);
        if ( s && (remaps[s] == 0) && (counts[s] >= low) )
        {
            remaps[s] = next_id;
            counts[next_id] = counts[s];
            ++next_id;
        }
    }

    counts.resize(next_id);

    std::ptrdiff_t xdim = seg_ptr->shape()[0];
    std::ptrdiff_t ydim = seg_ptr->shape()[1];
    std::ptrdiff_t zdim = seg_ptr->shape()[2];

    std::ptrdiff_t total = xdim * ydim * zdim;

    ID* seg_raw = seg_ptr->data();

    for ( std::ptrdiff_t idx = 0; idx < total; ++idx )
    {
        seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])];
    }

    std::cout << "Done with remapping, total: " << (next_id-1) << std::endl;

    region_graph<ID,F> new_rg;

    std::vector<std::set<ID>> in_rg(next_id);

    for ( auto& it: rg )
    {
        ID s1 = remaps[sets.find_set(std::get<1>(it))];
        ID s2 = remaps[sets.find_set(std::get<2>(it))];

        if ( s1 != s2 && s1 && s2 )
        {
            auto mm = std::minmax(s1,s2);
            if ( in_rg[mm.first].count(mm.second) == 0 )
            {
                new_rg.emplace_back(std::get<0>(it), mm.first, mm.second);
                in_rg[mm.first].insert(mm.second);
            }
        }
    }

    rg.swap(new_rg);

    std::cout << "Done with updating the region graph, size: "
              << rg.size() << std::endl;
}


template< typename ID, typename F, typename FN, typename M >
inline void merge_segments_with_function( const volume_ptr<ID>& seg_ptr,
                                          const region_graph_ptr<ID,F> rg_ptr,
                                          std::vector<std::size_t>& counts,
                                          const FN& func,
                                          const M& lowt,
                                           bool recreate_rg)
{
    zi::disjoint_sets<ID> sets(counts.size());

    region_graph<ID,F>& rg  = *rg_ptr;

    for ( auto& it: rg )
    {
        std::size_t size = func(std::get<0>(it));

        if ( size == 0 )
        {
            break;
        }

        ID s1 = sets.find_set(std::get<1>(it));
        ID s2 = sets.find_set(std::get<2>(it));

        if ( s1 != s2 && s1 && s2 )
        {
            if ( (counts[s1] < size) || (counts[s2] < size) )
            {
                counts[s1] += counts[s2];
                counts[s2]  = 0;
                ID s = sets.join(s1,s2);
                std::swap(counts[s], counts[s1]);
            }
        }
    }

    std::cout << "Done with merging" << std::endl;

    std::vector<ID> remaps(counts.size());

    ID next_id = 1;

    std::size_t low = static_cast<std::size_t>(lowt);

    for ( ID id = 0; id < counts.size(); ++id )
    {
        ID s = sets.find_set(id);
        if ( s && (remaps[s] == 0) && (counts[s] >= low) )
        {
            remaps[s] = next_id;
            counts[next_id] = counts[s];
            ++next_id;
        }
    }

    counts.resize(next_id);

    std::ptrdiff_t xdim = seg_ptr->shape()[0];
    std::ptrdiff_t ydim = seg_ptr->shape()[1];
    std::ptrdiff_t zdim = seg_ptr->shape()[2];

    std::ptrdiff_t total = xdim * ydim * zdim;

    ID* seg_raw = seg_ptr->data();

    for ( std::ptrdiff_t idx = 0; idx < total; ++idx )
    {
        seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])];
    }

    std::cout << "Done with remapping, total: " << (next_id-1) << std::endl;

    region_graph<ID,F> new_rg;

    std::vector<std::set<ID>> in_rg(next_id);

    for ( auto& it: rg )
    {
        ID s1 = remaps[sets.find_set(std::get<1>(it))];
        ID s2 = remaps[sets.find_set(std::get<2>(it))];

        if ( s1 != s2 && s1 && s2 )
        {
            auto mm = std::minmax(s1,s2);
            if ( in_rg[mm.first].count(mm.second) == 0 )
            {
                new_rg.push_back(std::make_tuple(std::get<0>(it), mm.first, mm.second));
                in_rg[mm.first].insert(mm.second);
            }
        }
    }

    if(recreate_rg)
        rg.swap(new_rg);

    std::cout << "Done with updating the region graph, size: "
              << rg.size() << std::endl;
}


template< typename ID, typename F, typename FN, typename M >
inline std::vector<double>
merge_segments_with_function_err( const volume_ptr<ID>& seg_ptr,
                                  const volume_ptr<ID>& gt_ptr,
                                  const region_graph_ptr<ID,F> rg_ptr,
                                  std::vector<std::size_t>& counts,
                                  const FN& func,
                                  const M& lowt )
{
    std::vector<double> ret;

    std::ptrdiff_t xdim = seg_ptr->shape()[0];
    std::ptrdiff_t ydim = seg_ptr->shape()[1];
    std::ptrdiff_t zdim = seg_ptr->shape()[2];

    volume<ID>& seg = *seg_ptr;
    volume<ID>& gt  = *gt_ptr;

    zi::disjoint_sets<ID> sets(counts.size());

    std::size_t tot = 0;

    std::vector<std::size_t> s_i(counts.size());
    std::map<ID,std::size_t> t_j;

    std::vector<std::map<ID,std::size_t>> p_ij(counts.size());

    for ( std::ptrdiff_t z = 0; z < zdim; ++z )
        for ( std::ptrdiff_t y = 0; y < ydim; ++y )
            for ( std::ptrdiff_t x = 0; x < xdim; ++x )
            {
                uint32_t sgv = seg[x][y][z];
                uint32_t gtv =  gt[x][y][z];

                if ( gtv )
                {
                    tot += 1;

                    ++p_ij[sgv][gtv];

                    ++s_i[sgv];
                    ++t_j[gtv];
                }
            }

    double sum_p_ij = 0;
    for ( auto& a: p_ij )
    {
        for ( auto& b: a )
        {
            sum_p_ij += (static_cast<double>(b.second) / tot) *
                (static_cast<double>(b.second) / tot);
        }
    }

    double sum_t_k = 0;
    for ( auto& a: t_j )
    {
        sum_t_k += (static_cast<double>(a.second) / tot) *
            (static_cast<double>(a.second) / tot);
    }


    double sum_s_k = 0;
    for ( auto& a: s_i )
    {
        sum_s_k += (static_cast<double>(a) / tot) *
            (static_cast<double>(a) / tot);
    }

    ret.push_back(sum_p_ij/sum_t_k);
    ret.push_back(sum_p_ij/sum_s_k);

    region_graph<ID,F>& rg  = *rg_ptr;

    int mod = 0;

    F minf = 0;

    for ( auto& it: rg )
    {
        std::size_t size = func(std::get<0>(it));

        if ( size == 0 )
        {
            minf = std::get<0>(it);
            break;
        }

        ID s1 = sets.find_set(std::get<1>(it));
        ID s2 = sets.find_set(std::get<2>(it));

        if ( s1 != s2 && s1 && s2 )
        {
            if ( (counts[s1] < size) || (counts[s2] < size) )
            {
                counts[s1] += counts[s2];
                counts[s2]  = 0;

                sum_s_k -= (static_cast<double>(s_i[s1])/tot) *
                    (static_cast<double>(s_i[s1])/tot);

                sum_s_k -= (static_cast<double>(s_i[s2])/tot) *
                    (static_cast<double>(s_i[s2])/tot);

                s_i[s1] += s_i[s2];
                s_i[s2]  = 0;

                sum_s_k += (static_cast<double>(s_i[s1])/tot) *
                    (static_cast<double>(s_i[s1])/tot);


                for ( auto& b: p_ij[s1] )
                {
                    sum_p_ij -= (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                }

                for ( auto& b: p_ij[s2] )
                {
                    sum_p_ij -= (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                    p_ij[s1][b.first] += b.second;
                }

                for ( auto& b: p_ij[s1] )
                {
                    sum_p_ij += (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                }

                p_ij[s2].clear();

                if ( (++mod) % 100 == 0 )
                {
                    //std::cout << "Now Error: " << (sum_p_ij/sum_t_k) << " "
                    //          << (sum_p_ij/sum_s_k) << "\n";
                    ret.push_back(sum_p_ij/sum_t_k);
                    ret.push_back(sum_p_ij/sum_s_k);
                }


                ID s = sets.join(s1,s2);
                std::swap(counts[s], counts[s1]);
                std::swap(s_i[s], s_i[s1]);
                std::swap(p_ij[s], p_ij[s1]);
            }
        }
    }

    std::cout << "Done with merging" << std::endl;


    for ( auto& it: rg )
    {
        break;

        if (  minf > std::get<0>(it) )
        {
            break;
        }


        ID s1 = sets.find_set(std::get<1>(it));
        ID s2 = sets.find_set(std::get<2>(it));

        if ( s1 != s2 && s1 && s2 )
        {
            //if ( (counts[s1] < 100) || (counts[s2] < 100) )
            {
                counts[s1] += counts[s2];
                counts[s2]  = 0;

                sum_s_k -= (static_cast<double>(s_i[s1])/tot) *
                    (static_cast<double>(s_i[s1])/tot);

                sum_s_k -= (static_cast<double>(s_i[s2])/tot) *
                    (static_cast<double>(s_i[s2])/tot);

                s_i[s1] += s_i[s2];
                s_i[s2]  = 0;

                sum_s_k += (static_cast<double>(s_i[s1])/tot) *
                    (static_cast<double>(s_i[s1])/tot);


                for ( auto& b: p_ij[s1] )
                {
                    sum_p_ij -= (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                }

                for ( auto& b: p_ij[s2] )
                {
                    sum_p_ij -= (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                    p_ij[s1][b.first] += b.second;
                }

                for ( auto& b: p_ij[s1] )
                {
                    sum_p_ij += (static_cast<double>(b.second) / tot) *
                        (static_cast<double>(b.second) / tot);
                }

                p_ij[s2].clear();

                //if ( (++mod) % 100 == 0 )
                std::cout << "Now Error: " << (sum_p_ij/sum_t_k) << " "
                          << (sum_p_ij/sum_s_k) << "\n";
                ret.push_back(sum_p_ij/sum_t_k);
                ret.push_back(sum_p_ij/sum_s_k);


                ID s = sets.join(s1,s2);
                std::swap(counts[s], counts[s1]);
                std::swap(s_i[s], s_i[s1]);
                std::swap(p_ij[s], p_ij[s1]);
            }
        }
    }

    std::cout << "Done with merging" << std::endl;


    std::vector<ID> remaps(counts.size());

    ID next_id = 1;

    std::size_t low = static_cast<std::size_t>(lowt);

    for ( ID id = 0; id < counts.size(); ++id )
    {
        ID s = sets.find_set(id);
        if ( s && (remaps[s] == 0) && (counts[s] >= low) )
        {
            remaps[s] = next_id;
            counts[next_id] = counts[s];
            ++next_id;
        }
    }

    counts.resize(next_id);

    std::ptrdiff_t total = xdim * ydim * zdim;

    ID* seg_raw = seg_ptr->data();

    for ( std::ptrdiff_t idx = 0; idx < total; ++idx )
    {
        seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])];
    }

    std::cout << "Done with remapping, total: " << (next_id-1) << std::endl;

    region_graph<ID,F> new_rg;

    std::vector<std::set<ID>> in_rg(next_id);

    for ( auto& it: rg )
    {
        ID s1 = remaps[sets.find_set(std::get<1>(it))];
        ID s2 = remaps[sets.find_set(std::get<2>(it))];

        if ( s1 != s2 && s1 && s2 )
        {
            auto mm = std::minmax(s1,s2);
            if ( in_rg[mm.first].count(mm.second) == 0 )
            {
                new_rg.emplace_back(std::get<0>(it), mm.first, mm.second);
                in_rg[mm.first].insert(mm.second);
            }
        }
    }

    rg.swap(new_rg);

    std::cout << "Done with updating the region graph, size: "
              << rg.size() << std::endl;

    return ret;
}
