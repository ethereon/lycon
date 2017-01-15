#pragma once

#include "lycon/defs.h"

namespace lycon
{

template <typename _Tp> class DataType
{
  public:
    typedef _Tp value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 1,
        depth = -1,
        channels = 1,
        fmt = 0,
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<bool>
{
  public:
    typedef bool value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_8U,
        channels = 1,
        fmt = (int)'u',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<uchar>
{
  public:
    typedef uchar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_8U,
        channels = 1,
        fmt = (int)'u',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<schar>
{
  public:
    typedef schar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_8S,
        channels = 1,
        fmt = (int)'c',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<char>
{
  public:
    typedef schar value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_8S,
        channels = 1,
        fmt = (int)'c',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<ushort>
{
  public:
    typedef ushort value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_16U,
        channels = 1,
        fmt = (int)'w',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<short>
{
  public:
    typedef short value_type;
    typedef int work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_16S,
        channels = 1,
        fmt = (int)'s',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<int>
{
  public:
    typedef int value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_32S,
        channels = 1,
        fmt = (int)'i',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<float>
{
  public:
    typedef float value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_32F,
        channels = 1,
        fmt = (int)'f',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <> class DataType<double>
{
  public:
    typedef double value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum
    {
        generic_type = 0,
        depth = LYCON_64F,
        channels = 1,
        fmt = (int)'d',
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <typename _Tp> class DataDepth
{
  public:
    enum
    {
        value = DataType<_Tp>::depth,
        fmt = DataType<_Tp>::fmt
    };
};

template <int _depth> class TypeDepth
{
    enum
    {
        depth = LYCON_USRTYPE1
    };
    typedef void value_type;
};

template <> class TypeDepth<LYCON_8U>
{
    enum
    {
        depth = LYCON_8U
    };
    typedef uchar value_type;
};

template <> class TypeDepth<LYCON_8S>
{
    enum
    {
        depth = LYCON_8S
    };
    typedef schar value_type;
};

template <> class TypeDepth<LYCON_16U>
{
    enum
    {
        depth = LYCON_16U
    };
    typedef ushort value_type;
};

template <> class TypeDepth<LYCON_16S>
{
    enum
    {
        depth = LYCON_16S
    };
    typedef short value_type;
};

template <> class TypeDepth<LYCON_32S>
{
    enum
    {
        depth = LYCON_32S
    };
    typedef int value_type;
};

template <> class TypeDepth<LYCON_32F>
{
    enum
    {
        depth = LYCON_32F
    };
    typedef float value_type;
};

template <> class TypeDepth<LYCON_64F>
{
    enum
    {
        depth = LYCON_64F
    };
    typedef double value_type;
};
}
