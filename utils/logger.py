# -*- coding: utf-8 -*-
# ************************************************************#
# FileName      : logger.py
# Objective     :
# Created by    :
# Created on    : 08/24/2020
# Last modified : 08/24/2020 15:24:18
# Description   :
#   V1.0 basic function
# ************************************************************#

import sys


class Logger:
    def __init__(self, filename = "default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == "__main__":
    sys.stdout = Logger("target_file.txt")
    print(111111111111111)
