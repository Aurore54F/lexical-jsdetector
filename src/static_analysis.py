# Copyright (C) 2019 Dennis Salzmann
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import os
import sys
from datetime import datetime
from subprocess import run, PIPE


def start_static_analysis(files, data, debug, verbose, string):
    path = sys.path[0]
    path = os.path.join(path, "jassi/src/jassi")
    for file in files:
        if verbose:
            print(str(datetime.now()) + ": Starting to analyse " + file.path + " for " + string)
        lexed = run([path, file.path], stdout=PIPE)
        if verbose:
            print(str(datetime.now()) + ": Finished to analyse " + file.path + " for " + string + " with exit code "
                  + str(lexed.returncode))
        if lexed.returncode == 0:
            data.append(json.dumps({
                'type': 'static',
                'file': file.path,
                'parent_html': file.parent_html,
                'classification': file.classification,
                'return_code': lexed.returncode,
                'output': (lexed.stdout.decode('utf-8')).replace("\n", "")
            }))
    return
