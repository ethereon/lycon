#pragma once

#include <cstddef>
#include <cstring>
#include <string>

#include "lycon/util/macros.h"

namespace lycon
{

class FileNode;

class String
{
  public:
    typedef char value_type;
    typedef char &reference;
    typedef const char &const_reference;
    typedef char *pointer;
    typedef const char *const_pointer;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef char *iterator;
    typedef const char *const_iterator;

    static const size_t npos = size_t(-1);

    explicit String();
    String(const String &str);
    String(const String &str, size_t pos, size_t len = npos);
    String(const char *s);
    String(const char *s, size_t n);
    String(size_t n, char c);
    String(const char *first, const char *last);
    template <typename Iterator> String(Iterator first, Iterator last);
    explicit String(const FileNode &fn);
    ~String();

    String &operator=(const String &str);
    String &operator=(const char *s);
    String &operator=(char c);

    String &operator+=(const String &str);
    String &operator+=(const char *s);
    String &operator+=(char c);

    size_t size() const;
    size_t length() const;

    char operator[](size_t idx) const;
    char operator[](int idx) const;

    const char *begin() const;
    const char *end() const;

    const char *c_str() const;

    bool empty() const;
    void clear();

    int compare(const char *s) const;
    int compare(const String &str) const;

    void swap(String &str);
    String substr(size_t pos = 0, size_t len = npos) const;

    size_t find(const char *s, size_t pos, size_t n) const;
    size_t find(char c, size_t pos = 0) const;
    size_t find(const String &str, size_t pos = 0) const;
    size_t find(const char *s, size_t pos = 0) const;

    size_t rfind(const char *s, size_t pos, size_t n) const;
    size_t rfind(char c, size_t pos = npos) const;
    size_t rfind(const String &str, size_t pos = npos) const;
    size_t rfind(const char *s, size_t pos = npos) const;

    size_t find_first_of(const char *s, size_t pos, size_t n) const;
    size_t find_first_of(char c, size_t pos = 0) const;
    size_t find_first_of(const String &str, size_t pos = 0) const;
    size_t find_first_of(const char *s, size_t pos = 0) const;

    size_t find_last_of(const char *s, size_t pos, size_t n) const;
    size_t find_last_of(char c, size_t pos = npos) const;
    size_t find_last_of(const String &str, size_t pos = npos) const;
    size_t find_last_of(const char *s, size_t pos = npos) const;

    friend String operator+(const String &lhs, const String &rhs);
    friend String operator+(const String &lhs, const char *rhs);
    friend String operator+(const char *lhs, const String &rhs);
    friend String operator+(const String &lhs, char rhs);
    friend String operator+(char lhs, const String &rhs);

    String toLowerCase() const;

    String(const std::string &str);
    String(const std::string &str, size_t pos, size_t len = npos);
    String &operator=(const std::string &str);
    String &operator+=(const std::string &str);
    operator std::string() const;

    friend String operator+(const String &lhs, const std::string &rhs);
    friend String operator+(const std::string &lhs, const String &rhs);

  private:
    char *cstr_;
    size_t len_;

    char *allocate(size_t len); // len without trailing 0
    void deallocate();

    String(int); // disabled and invalid. Catch invalid usages like, commandLineParser.has(0) problem
};

inline String::String() : cstr_(0), len_(0)
{
}

inline String::String(const String &str) : cstr_(str.cstr_), len_(str.len_)
{
    if (cstr_)
        LYCON_XADD(((int *)cstr_) - 1, 1);
}

inline String::String(const String &str, size_t pos, size_t len) : cstr_(0), len_(0)
{
    pos = std::min(pos, str.len_);
    len = std::min(str.len_ - pos, len);
    if (!len)
        return;
    if (len == str.len_)
    {
        LYCON_XADD(((int *)str.cstr_) - 1, 1);
        cstr_ = str.cstr_;
        len_ = str.len_;
        return;
    }
    memcpy(allocate(len), str.cstr_ + pos, len);
}

inline String::String(const char *s) : cstr_(0), len_(0)
{
    if (!s)
        return;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
}

inline String::String(const char *s, size_t n) : cstr_(0), len_(0)
{
    if (!n)
        return;
    memcpy(allocate(n), s, n);
}

inline String::String(size_t n, char c) : cstr_(0), len_(0)
{
    memset(allocate(n), c, n);
}

inline String::String(const char *first, const char *last) : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    memcpy(allocate(len), first, len);
}

template <typename Iterator> inline String::String(Iterator first, Iterator last) : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    char *str = allocate(len);
    while (first != last)
    {
        *str++ = *first;
        ++first;
    }
}

inline String::~String()
{
    deallocate();
}

inline String &String::operator=(const String &str)
{
    if (&str == this)
        return *this;

    deallocate();
    if (str.cstr_)
        LYCON_XADD(((int *)str.cstr_) - 1, 1);
    cstr_ = str.cstr_;
    len_ = str.len_;
    return *this;
}

inline String &String::operator=(const char *s)
{
    deallocate();
    if (!s)
        return *this;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
    return *this;
}

inline String &String::operator=(char c)
{
    deallocate();
    allocate(1)[0] = c;
    return *this;
}

inline String &String::operator+=(const String &str)
{
    *this = *this + str;
    return *this;
}

inline String &String::operator+=(const char *s)
{
    *this = *this + s;
    return *this;
}

inline String &String::operator+=(char c)
{
    *this = *this + c;
    return *this;
}

inline size_t String::size() const
{
    return len_;
}

inline size_t String::length() const
{
    return len_;
}

inline char String::operator[](size_t idx) const
{
    return cstr_[idx];
}

inline char String::operator[](int idx) const
{
    return cstr_[idx];
}

inline const char *String::begin() const
{
    return cstr_;
}

inline const char *String::end() const
{
    return len_ ? cstr_ + 1 : 0;
}

inline bool String::empty() const
{
    return len_ == 0;
}

inline const char *String::c_str() const
{
    return cstr_ ? cstr_ : "";
}

inline void String::swap(String &str)
{
    std::swap(cstr_, str.cstr_);
    std::swap(len_, str.len_);
}

inline void String::clear()
{
    deallocate();
}

inline int String::compare(const char *s) const
{
    if (cstr_ == s)
        return 0;
    return strcmp(c_str(), s);
}

inline int String::compare(const String &str) const
{
    if (cstr_ == str.cstr_)
        return 0;
    return strcmp(c_str(), str.c_str());
}

inline String String::substr(size_t pos, size_t len) const
{
    return String(*this, pos, len);
}

inline size_t String::find(const char *s, size_t pos, size_t n) const
{
    if (n == 0 || pos + n > len_)
        return npos;
    const char *lmax = cstr_ + len_ - n;
    for (const char *i = cstr_ + pos; i <= lmax; ++i)
    {
        size_t j = 0;
        while (j < n && s[j] == i[j])
            ++j;
        if (j == n)
            return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::find(char c, size_t pos) const
{
    return find(&c, pos, 1);
}

inline size_t String::find(const String &str, size_t pos) const
{
    return find(str.c_str(), pos, str.len_);
}

inline size_t String::find(const char *s, size_t pos) const
{
    if (pos >= len_ || !s[0])
        return npos;
    const char *lmax = cstr_ + len_;
    for (const char *i = cstr_ + pos; i < lmax; ++i)
    {
        size_t j = 0;
        while (s[j] && s[j] == i[j])
        {
            if (i + j >= lmax)
                return npos;
            ++j;
        }
        if (!s[j])
            return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::rfind(const char *s, size_t pos, size_t n) const
{
    if (n > len_)
        return npos;
    if (pos > len_ - n)
        pos = len_ - n;
    for (const char *i = cstr_ + pos; i >= cstr_; --i)
    {
        size_t j = 0;
        while (j < n && s[j] == i[j])
            ++j;
        if (j == n)
            return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::rfind(char c, size_t pos) const
{
    return rfind(&c, pos, 1);
}

inline size_t String::rfind(const String &str, size_t pos) const
{
    return rfind(str.c_str(), pos, str.len_);
}

inline size_t String::rfind(const char *s, size_t pos) const
{
    return rfind(s, pos, strlen(s));
}

inline size_t String::find_first_of(const char *s, size_t pos, size_t n) const
{
    if (n == 0 || pos + n > len_)
        return npos;
    const char *lmax = cstr_ + len_;
    for (const char *i = cstr_ + pos; i < lmax; ++i)
    {
        for (size_t j = 0; j < n; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::find_first_of(char c, size_t pos) const
{
    return find_first_of(&c, pos, 1);
}

inline size_t String::find_first_of(const String &str, size_t pos) const
{
    return find_first_of(str.c_str(), pos, str.len_);
}

inline size_t String::find_first_of(const char *s, size_t pos) const
{
    if (len_ == 0)
        return npos;
    if (pos >= len_ || !s[0])
        return npos;
    const char *lmax = cstr_ + len_;
    for (const char *i = cstr_ + pos; i < lmax; ++i)
    {
        for (size_t j = 0; s[j]; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::find_last_of(const char *s, size_t pos, size_t n) const
{
    if (len_ == 0)
        return npos;
    if (pos >= len_)
        pos = len_ - 1;
    for (const char *i = cstr_ + pos; i >= cstr_; --i)
    {
        for (size_t j = 0; j < n; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline size_t String::find_last_of(char c, size_t pos) const
{
    return find_last_of(&c, pos, 1);
}

inline size_t String::find_last_of(const String &str, size_t pos) const
{
    return find_last_of(str.c_str(), pos, str.len_);
}

inline size_t String::find_last_of(const char *s, size_t pos) const
{
    if (len_ == 0)
        return npos;
    if (pos >= len_)
        pos = len_ - 1;
    for (const char *i = cstr_ + pos; i >= cstr_; --i)
    {
        for (size_t j = 0; s[j]; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline String String::toLowerCase() const
{
    String res(cstr_, len_);

    for (size_t i = 0; i < len_; ++i)
        res.cstr_[i] = (char)::tolower(cstr_[i]);

    return res;
}

inline String operator+(const String &lhs, const String &rhs)
{
    String s;
    s.allocate(lhs.len_ + rhs.len_);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.cstr_, rhs.len_);
    return s;
}

inline String operator+(const String &lhs, const char *rhs)
{
    String s;
    size_t rhslen = strlen(rhs);
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs, rhslen);
    return s;
}

inline String operator+(const char *lhs, const String &rhs)
{
    String s;
    size_t lhslen = strlen(lhs);
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs, lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline String operator+(const String &lhs, char rhs)
{
    String s;
    s.allocate(lhs.len_ + 1);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    s.cstr_[lhs.len_] = rhs;
    return s;
}

inline String operator+(char lhs, const String &rhs)
{
    String s;
    s.allocate(rhs.len_ + 1);
    s.cstr_[0] = lhs;
    memcpy(s.cstr_ + 1, rhs.cstr_, rhs.len_);
    return s;
}

static inline bool operator==(const String &lhs, const String &rhs)
{
    return 0 == lhs.compare(rhs);
}
static inline bool operator==(const char *lhs, const String &rhs)
{
    return 0 == rhs.compare(lhs);
}
static inline bool operator==(const String &lhs, const char *rhs)
{
    return 0 == lhs.compare(rhs);
}
static inline bool operator!=(const String &lhs, const String &rhs)
{
    return 0 != lhs.compare(rhs);
}
static inline bool operator!=(const char *lhs, const String &rhs)
{
    return 0 != rhs.compare(lhs);
}
static inline bool operator!=(const String &lhs, const char *rhs)
{
    return 0 != lhs.compare(rhs);
}
static inline bool operator<(const String &lhs, const String &rhs)
{
    return lhs.compare(rhs) < 0;
}
static inline bool operator<(const char *lhs, const String &rhs)
{
    return rhs.compare(lhs) > 0;
}
static inline bool operator<(const String &lhs, const char *rhs)
{
    return lhs.compare(rhs) < 0;
}
static inline bool operator<=(const String &lhs, const String &rhs)
{
    return lhs.compare(rhs) <= 0;
}
static inline bool operator<=(const char *lhs, const String &rhs)
{
    return rhs.compare(lhs) >= 0;
}
static inline bool operator<=(const String &lhs, const char *rhs)
{
    return lhs.compare(rhs) <= 0;
}
static inline bool operator>(const String &lhs, const String &rhs)
{
    return lhs.compare(rhs) > 0;
}
static inline bool operator>(const char *lhs, const String &rhs)
{
    return rhs.compare(lhs) < 0;
}
static inline bool operator>(const String &lhs, const char *rhs)
{
    return lhs.compare(rhs) > 0;
}
static inline bool operator>=(const String &lhs, const String &rhs)
{
    return lhs.compare(rhs) >= 0;
}
static inline bool operator>=(const char *lhs, const String &rhs)
{
    return rhs.compare(lhs) <= 0;
}
static inline bool operator>=(const String &lhs, const char *rhs)
{
    return lhs.compare(rhs) >= 0;
}

inline String::String(const std::string &str) : cstr_(0), len_(0)
{
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
}

inline String::String(const std::string &str, size_t pos, size_t len) : cstr_(0), len_(0)
{
    size_t strlen = str.size();
    pos = std::min(pos, strlen);
    len = std::min(strlen - pos, len);
    if (!len)
        return;
    memcpy(allocate(len), str.c_str() + pos, len);
}

inline String &String::operator=(const std::string &str)
{
    deallocate();
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
    return *this;
}

inline String &String::operator+=(const std::string &str)
{
    *this = *this + str;
    return *this;
}

inline String::operator std::string() const
{
    return std::string(cstr_, len_);
}

inline String operator+(const String &lhs, const std::string &rhs)
{
    String s;
    size_t rhslen = rhs.size();
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.c_str(), rhslen);
    return s;
}

inline String operator+(const std::string &lhs, const String &rhs)
{
    String s;
    size_t lhslen = lhs.size();
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs.c_str(), lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}
}
