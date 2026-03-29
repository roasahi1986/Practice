// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/os/network.h"

#ifdef __APPLE__

#include <unistd.h>      // gethostname
#include <netdb.h>       // gethostbyname
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <string>

bool get_host_ip(std::string *ip) {
  if (nullptr == ip) {
    return false;
  }

  char hostname[256];
  if (0 != gethostname(hostname, sizeof(hostname))) {
    return false;
  }

  struct hostent *host_entry = gethostbyname(hostname);
  if (nullptr == host_entry) {
    return false;
  }

  *ip = inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[0]));

  return true;
}

#elif defined(__linux__)

#include <unistd.h>      // gethostname
#include <netdb.h>       // gethostbyname
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <string>

bool get_host_ip(std::string *ip) {
  if (nullptr == ip) {
    return false;
  }

  char hostname[256];
  if (0 != gethostname(hostname, sizeof(hostname))) {
    return false;
  }

  struct hostent *host_entry = gethostbyname(hostname);
  if (nullptr == host_entry) {
    return false;
  }

  *ip = inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[0]));

  return true;
}

#endif
