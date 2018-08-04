// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INTERNAL_SAPIEN_UTILITY_BYTE_ORDERER_H_
#define INTERNAL_SAPIEN_UTILITY_BYTE_ORDERER_H_

#include "sapien/internal/port.h"
#include "sapien/internal/type_traits/is_pointer.h"

namespace sapien {
namespace internal {

class ByteOrderer {
 public:
  // Default constructor
  ByteOrderer();

  // Determine the endianess of the host machine
  bool HostIsBigEndian() const { return !little_endian_; }
  bool HostIsLittleEndian() const { return little_endian_; }

  // Converts bytes from host machine (i.e the machine that is currently
  // executing this method) to big endian ordering.
  // If the host machine is also a big endian machine, this method
  // simply does nothing!
  template<typename DataType>
  void HostToBigEndian(DataType* data) const {
    if (little_endian_) { FlipBytes_(data); }
  }

  // Converts bytes from big endian ordring to the host machine.
  // For example, when we donwload some data from the web, and the piece of
  // data happened to be in big endian ordering, and our machine is
  // little endian => we need to convert it to little endian before using it.
  template<typename DataType>
  void BigEndianToHost(DataType* data) const {
    if (little_endian_) { FlipBytes_(data); }
  }

  // Converts bytes from host to little endian ordering
  template<typename DataType>
  void HostToLittleEndian(DataType* data) const {
    if (!little_endian_) { FlipBytes_(data); }
  }

  // Coverts bytes from little endian ordering to host.
  template<typename DataType>
  void LittleEndianToHost(DataType* data) const {
    if (!little_endian_) { FlipBytes_(data); }
  }

 private:
  bool little_endian_;

  template<typename DataType>
  void FlipBytes_(DataType* data) const;
};

// Implementation.

ByteOrderer::ByteOrderer() {
  unsigned int temp = 1;
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&temp);
  little_endian_ = (*ptr == 1);
}

template<typename DataType>
void ByteOrderer::FlipBytes_(DataType* data) const {
  // Ignore if data type is pointer.
  if (is_pointer<DataType>::value) { return; }

  const size_t n_bytes = sizeof(DataType);
  uint8_t* ptr = reinterpret_cast<uint8_t*>(data);

  if (n_bytes > 1) {
    size_t i = 0;
    size_t j = n_bytes - 1;
    uint8_t c;

    while (i < j) {
      c = ptr[i];
      ptr[i] = ptr[j];
      ptr[j] = c;
      i++;
      j--;
    }
  }
}

}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_BYTE_ORDERER_H_
