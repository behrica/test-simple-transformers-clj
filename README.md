# test-simple-transformers-clj

An example of using the clojure python binding library
libpython-clj

https://github.com/clj-python/libpython-clj

with the python based simpletransformers library.

This combination offers the fastest route to use Bert and other deeplearning models from Clojure.


https://github.com/ThilinaRajapakse/simpletransformers

This repo contains a Dockerfile which installs :
- clojure
- anaconda python
- simpletransformers and all its dependencies

The clojure code is the one-to-one translation of the minimal start for binary classification with the RoBerta DL model.


https://github.com/ThilinaRajapakse/simpletransformers#minimal-start-for-binary-classification


## License

Copyright © 2020 Carsten Behring

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
