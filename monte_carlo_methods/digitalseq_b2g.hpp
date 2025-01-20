#ifndef _DIGITALSEQ_B2G_HPP
#define _DIGITALSEQ_B2G_HPP

////
// (C) Dirk Nuyens, KU Leuven, 2014,2015,2016,...

#include <cstdint>
#include <limits>
#include <vector>
#include <cassert>

namespace qmc {

    /// \brief Count the number of trailing zero bits.
    /// 
    /// \see Count the consecutive zero bits (trailing) on the right in parallel
    /// http://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
    inline unsigned count_trailing_zero_bits(std::uint32_t v)
    {
        unsigned c = 32;
        v &= -signed(v);
        if (v) c--;
        if (v & 0x0000FFFF) c -= 16;
        if (v & 0x00FF00FF) c -= 8;
        if (v & 0x0F0F0F0F) c -= 4;
        if (v & 0x33333333) c -= 2;
        if (v & 0x55555555) c -= 1;
        return c;
    }

    /// \brief Reverse the bits in a std::uint32_t.
    /// 
    /// \see Bit reverse code from Stanford Bit hacks page:
    /// https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
    std::uint32_t bitreverse(std::uint32_t k)
    {
        std::uint32_t v = k;
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);  // swap odd and even bits
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);  // swap consecutive pairs 
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);  // swap nibbles ... 
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);  // swap bytes
        v = ( v >> 16             ) | ( v               << 16); // swap 2-byte long pairs
        return v;
    }

    /// \brief Reverse the bits in a std::uint64_t.
    /// Calls twice the version for std::uint32_t.
    /// \see bitreverse(std::uint32_t)
    std::uint64_t bitreverse(std::uint64_t k)
    {
        return (std::uint64_t(bitreverse(std::uint32_t(k))) << 32) | std::uint64_t(bitreverse(std::uint32_t(k >> 32)));
    }

    /// \brief Base 2 digital sequence point generator (in gray coded radical inverse ordering).
    template <typename FLOAT_T=long double, typename UINT_T=std::uint64_t>
    struct digitalseq_b2g
    {
        typedef FLOAT_T float_t;
        typedef UINT_T  uint_t;
        enum init_t { INIT0, INIT1, INITSKIP };

        // TODO: add static_assert for base 2 types?
        static const int uint_t_digits2  = std::numeric_limits<uint_t>::digits;     // assuming this is base 2 digits
        static const int float_t_digits2 = std::numeric_limits<float_t>::digits;    // assuming this is base 2 digits
        static constexpr const float_t recipd = 1 / float_t(uint_t(1) << (uint_t_digits2-1)) / 2; // assuming base 2 types

        std::uint32_t k;            //< index of the next point
        const unsigned m;           //< this base-2-digital-sequence has n=2^m points
        const unsigned s;           //< number of generating matrices, i.e., number of dimensions
        const uint_t n;             //< maximum number of points n=2^m
        const init_t initmode;      //< what to do with the first point of the sequence (the zero point)
        std::vector<uint_t> Cs;     //< s generating matrices consisting of m integers encoding the columns with lsb in the top row
        std::vector<uint_t> cur;    //< s dimensional vector of current point as binary number completely shifted to the left
        std::vector<float_t> x;     //< the current point converted to float, x = cur * recipd

        /// Constructor.
        ///
        /// \param s            number of dimensions
        /// \param m            number of points is 2^m
        /// \param Cs_begin     iterator to the generating matrices, they are copied into this object
        /// \param initmode     one of INIT0: first point is (0, ..., 0)
        ///                            INIT1: first point is (1, ..., 1)
        ///                            INITSKIP: the first point is skipped
        ///
        /// The generating matrices are stored as a flat list of s \times m
        /// integers which represent the columns of the generating matrices
        /// stored as integers with the lsb in the top row.
        ///
        /// The first output point of this sequence (depending on initmode)
        /// will have been calculated and is immediately available as a floating
        /// point vector of type std::vector<float_t> as the member variable x,
        /// or by using the dereference operator on this object. The vector can
        /// be obtained as an integer aligned as far to the left as possible as
        /// the member variable cur.
        ///
        /// The sequence can be brought back into the initial state by calling
        /// reset(), which will calculate the first point again and set the
        /// appropriate state.
        /// 
        /// Skipping to an arbitrary point in the sequence is possible by
        /// calling the set_state(std::uint32_t) method.
        ///
        /// Advancing to the next point in the sequence is done by the next()
        /// method.
        ///
        /// \see x, operator*(), cur, reset(), set_state(std::uint32_t), next()
        template <typename InputIterator>
        digitalseq_b2g(unsigned s, unsigned m, InputIterator Cs_begin, init_t initmode=INIT0)
        : k(initmode != INITSKIP ? 0 : 1), m(m), s(s), n(uint_t(1) << m), initmode(initmode), Cs(Cs_begin, Cs_begin+m*s), cur(s), x(s)
        {
            assert(m < 32);
            assert(std::numeric_limits<uint_t>::radix == 2 &&       // this will guarantee full precision, but we don't check
                   std::numeric_limits<float_t>::radix == 2 &&      // if both types are not base 2
                   float_t_digits2 >= uint_t_digits2);              // (and asserts are also not checked if you compile with -DNDEBUG)
            for(auto& a : Cs) a = bitreverse(a);
            reset();
        }

        /// Advance to the next point in the sequence.
        /// No checks are made if the sequence is already past the end.
        /// \see past_end to check if all points have been generated
        float_t* next()
        {
            unsigned ctz = count_trailing_zero_bits(k++); // figure out which bit changed in gray code ordering, and post increment k
            for(auto j = 0; j < s; ++j) {                 // then update the point by adding in that column of the generating matrix
                cur[j] ^= Cs[j*m+ctz];
                x[j] = recipd * cur[j];
            }
            return &x[0];
        }

        /// Advance to the next point in the sequence.
        /// This one is here to be able to use this class as a forward iterator.
        digitalseq_b2g& operator++()
        {
            next();
            return *this;
        }

        /// Return the current point as a floating point vector.
        /// This one is here to be able to use this class as a forward iterator.
        const std::vector<float_t>& operator*() const
        {
            return x;
        }

        /// Skip to the k-th point in the sequence.
        void set_state(std::uint32_t new_k)
        {
            if(new_k == 0) {
                reset();
                return;
            }
            std::fill(cur.begin(), cur.end(), 0);
            k = new_k;
            for(auto j = 0; j < s; ++j) {
                for(int i = 0; i < m; ++i) {
                    cur[j] ^= Cs[j*m+i];
                }
                x[j] = recipd * cur[j];
            }
            k++;
        }

        /// Reset the sequence to the initial state of the object (depending on initmode).
        void reset()
        {
            k = 1; // index of next point
            std::fill(cur.begin(), cur.end(), 0);
            if(initmode == INIT0) std::fill(x.begin(), x.end(), 0);
            else if(initmode == INIT1) std::fill(x.begin(), x.end(), 1);
            else if(initmode == INITSKIP) next();
        }

        /// Check if we have passed the end of the sequence.
        bool past_end() const
        {
            return k > n;
        }

        struct digitalseq_b2g_dummy_end
        {
        };

        digitalseq_b2g_dummy_end end() const
        {
            return digitalseq_b2g_dummy_end();
        }

        bool operator!=(const digitalseq_b2g_dummy_end&) const
        {
            return !past_end();
        }

    };

}

#endif // _DIGITALSEQ_B2G_HPP
