// Copyright 2018

#include <string>
#include <fstream>

#include "sapien/dataset.h"
#include "sapien/utility/byte_orderer.h"
#include "sapien/utility/stringprintf.h"
#include "glog/logging.h"

namespace sapien {

void LoadMNIST(const char* mnist_root_dir,
               uint8_t* X_train,
               uint8_t* y_train,
               uint8_t* X_test,
               uint8_t* y_test) {
  using internal::StringPrintf;
  using internal::ByteOrderer;

  std::string train_images_path, train_labels_path, test_images_path,
      test_labels_path;

  // Construct the absolute path to each file in the mnist directory.
  train_images_path = StringPrintf("%s/train-images-idx3-ubyte",
                                   mnist_root_dir);
  train_labels_path = StringPrintf("%s/train-labels-idx1-ubyte",
                                   mnist_root_dir);
  test_images_path = StringPrintf("%s/t10k-images-idx3-ubyte",
                                  mnist_root_dir);
  test_labels_path = StringPrintf("%s/t10k-labels-idx1-ubyte",
                                  mnist_root_dir);

  // Open binary files.
  std::ifstream train_images_file(train_images_path.c_str(), std::ios::binary);
  std::ifstream train_labels_file(train_labels_path.c_str(), std::ios::binary);
  std::ifstream test_images_file(test_images_path.c_str(), std::ios::binary);
  std::ifstream test_labels_file(test_labels_path.c_str(), std::ios::binary);

  // Check for potential opening errors.
  if (!train_images_file) {
    LOG(ERROR) << "Failed to open file: " << train_images_path;
    return;
  } else if (!train_labels_file) {
    LOG(ERROR) << "Failed to open file: " << train_labels_path;
    return;
  } else if (!test_images_file) {
    LOG(ERROR) << "Failed to open file: " << test_images_path;
    return;
  } else if (!test_labels_file) {
    LOG(ERROR) << "Failed to open file: " << test_labels_path;
  }

  ByteOrderer byte_orderer;

  //! Some header numbers from MNIST dataset as described in
  //! http://yann.lecun.com/exdb/mnist/
  uint32_t magic_number;
  uint32_t n_train_images, n_train_labels;
  uint32_t n_test_images, n_test_labels;
  uint32_t n_rows, n_cols;

  //! Read header numbers from train images file
  train_images_file.read(reinterpret_cast<char*>(&magic_number),
                         sizeof(magic_number));
  byte_orderer.BigEndianToHost(&magic_number);
  train_images_file.read(reinterpret_cast<char*>(&n_train_images),
                         sizeof(n_train_images));
  byte_orderer.BigEndianToHost(&n_train_images);
  train_images_file.read(reinterpret_cast<char*>(&n_rows),
                         sizeof(n_rows));
  byte_orderer.BigEndianToHost(&n_rows);
  train_images_file.read(reinterpret_cast<char*>(&n_cols),
                         sizeof(n_cols));
  byte_orderer.BigEndianToHost(&n_cols);

  if (magic_number != 2051 || n_train_images != 60000 ||
      n_rows != 28 || n_cols != 28) {
    LOG(ERROR) << "Wrong header format: " << train_images_path;
    return;
  }

  //! Read header numbers from train labels file
  train_labels_file.read(reinterpret_cast<char*>(&magic_number),
                         sizeof(magic_number));
  byte_orderer.BigEndianToHost(&magic_number);
  train_labels_file.read(reinterpret_cast<char*>(&n_train_images),
                         sizeof(n_train_images));
  byte_orderer.BigEndianToHost(&n_train_images);
  if (magic_number != 2049 || n_train_images != 60000) {
    LOG(ERROR) << "Wrong header format: " << train_labels_path;
    return;
  }

  //! Read header numbers from test images file
  test_images_file.read(reinterpret_cast<char*>(&magic_number),
                        sizeof(magic_number));
  byte_orderer.BigEndianToHost(&magic_number);
  test_images_file.read(reinterpret_cast<char*>(&n_test_images),
                        sizeof(n_test_images));
  byte_orderer.BigEndianToHost(&n_test_images);
  test_images_file.read(reinterpret_cast<char*>(&n_rows),
                        sizeof(n_rows));
  byte_orderer.BigEndianToHost(&n_rows);
  test_images_file.read(reinterpret_cast<char*>(&n_cols),
                        sizeof(n_cols));
  byte_orderer.BigEndianToHost(&n_cols);
  if (magic_number != 2051 || n_test_images != 10000 ||
      n_rows != 28 || n_cols != 28) {
    LOG(ERROR) << "Wrong header format: " << test_images_path;
    return;
  }

  //! Last but not least, read the header numbers from test labels file
  test_labels_file.read(reinterpret_cast<char*>(&magic_number),
                        sizeof(magic_number));
  byte_orderer.BigEndianToHost(&magic_number);
  test_labels_file.read(reinterpret_cast<char*>(&n_test_images),
                        sizeof(n_test_images));
  byte_orderer.BigEndianToHost(&n_test_images);
  if (magic_number != 2049 || n_test_images != 10000) {
    LOG(ERROR) << "Wrong header format: " << test_labels_path;
    return;
  }

  //! Check for errors again?
  if (!train_images_file) {
    LOG(ERROR) << "Failed to read: " << train_images_path;
    return;
  } else if (!train_labels_file) {
    LOG(ERROR) << "Failed to read: " << train_labels_path;
    return;
  } else if (!test_images_file) {
    LOG(ERROR) << "Failed to read: " << test_images_path;
    return;
  } else if (!test_labels_file) {
    LOG(ERROR) << "Failed to read: " << test_labels_path;
    return;
  }

  // 60000 images, each is 28x28 pixels.
  const size_t n_features = 28 * 28;
  const size_t n_train_samples = 60000;
  const size_t n_test_samples = 10000;

  train_images_file.read(reinterpret_cast<char*>(X_train),
                         n_train_images * n_features);
  train_labels_file.read(reinterpret_cast<char*>(y_train),
                         n_train_images);
  test_images_file.read(reinterpret_cast<char*>(X_test),
                        n_test_images * n_features);
  test_labels_file.read(reinterpret_cast<char*>(y_test),
                        n_test_images);

  // Last check
  if (train_images_file.get() != EOF) {
    LOG(ERROR) << "Unexpected bytes at the end of: "
               << train_images_path;
    return;
  } else if (train_labels_file.get() != EOF) {
    LOG(ERROR) << "Upexpected bytes at the end of: "
               << train_labels_path;
    return;
  } else if (test_images_file.get() != EOF) {
    LOG(ERROR) << "Unexpected bytes at the end of: "
               << test_images_path;
    return;
  } else if (test_labels_file.get() != EOF) {
    LOG(ERROR) << "Unexpected bytes at the end of: "
               << test_labels_path;
    return;
  }
}
}  // namespace sapien
