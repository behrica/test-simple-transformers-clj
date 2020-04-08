(ns test-simple-transformers-clj.core
  (:require 
   [libpython-clj.python   :refer [py. py.. py.-
                                   as-python as-jvm
                                   ->python ->jvm
                                   get-attr call-attr call-attr-kw
                                   get-item att-type-map
                                   call call-kw initialize!
                                   as-numpy as-tensor ->numpy
                                   run-simple-string
                                   add-module module-dict
                                   import-module
                                   python-type]
    :as py]
   )
  )

;; This points to python executable and library as it is created by the Dockerfile
(py/initialize! :python-executable "/root/miniconda3/envs/transformers/bin/python"
                :library-path "/root/miniconda3/envs/transformers/lib/libpython3.8.so"
                :no-io-redirect? true   ;without this, the python logging is not shown
                )

(require '[libpython-clj.require :refer [require-python]])

(require-python '[simpletransformers.classification :refer [ClassificationModel]])
(require-python '[pandas :as pd])
(require-python '[logging])


(logging/basicConfig :level logging/DEBUG)
(def transformers-logger (logging/getLogger "transformers"))
(py. transformers-logger setLevel logging/DEBUG)



(def train-data  [["Example sentence belonging to class 1"  1]
                  ["Example sentence belonging to class 0"  0]])

(def eval-data [["Example eval sentence belonging to class 1"  1]
                ["Example eval sentence belonging to class 0" 0]])



(def eval-df  (pd/DataFrame eval-data))
(def train-df  (pd/DataFrame train-data))

(def model (ClassificationModel "roberta"  "roberta-base" :use_cuda false
                                :args {:overwrite_output_dir true
                                       :use_multiprocessing false        ;; important for libpython-clj
                                       } ))
(py. model train_model train-df)
(py. model eval_model eval-df)

