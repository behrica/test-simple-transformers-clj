(println (java.util.Date.) "start")
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
(println (java.util.Date.) "0")

(py/initialize! :python-executable "/root/miniconda3/envs/transformers/bin/python"
                :library-path "/root/miniconda3/envs/transformers/lib/libpython3.8.so"
                :no-io-redirect? true
                )

                                        ;(py/initialize! :python-executable "/home/carsten/.conda/envs/transformers-cud10.2/bin/python3.6"
                                        ;                :library-path "/home/carsten/.conda/envs/transformers-cud10.2/lib/libpython3.6m.so")
(println (java.util.Date.) "1")

(require '[libpython-clj.require :refer [require-python]])

(require-python '[simpletransformers.classification :refer [ClassificationModel]])
(require-python '[pandas :as pd])
(require-python '[logging])


(println (java.util.Date.) "2")


(logging/basicConfig :level logging/DEBUG)
(def transformers-logger (logging/getLogger "transformers"))
(py. transformers-logger setLevel logging/DEBUG)

(println (java.util.Date.) "3")



(def train-data  [["Example sentence belonging to class 1"  1]
                  ["Example sentence belonging to class 0"  0]])

(def eval-data [["Example eval sentence belonging to class 1"  1]
                ["Example eval sentence belonging to class 0" 0]])

(println (java.util.Date.) "4")


(def eval-df  (pd/DataFrame eval-data))
(def train-df  (pd/DataFrame train-data))

(def n "single")
                                        ;(dotimes [n 10]
;(println "loop " n)
(println (java.util.Date.) n " --- " "5 - before ClassificationModel")
(def model (ClassificationModel "roberta"  "roberta-base" :use_cuda false
                                :args {:overwrite_output_dir true} ))

(println (java.util.Date.) n " --- " "6 - before train_model")
(py. model train_model train-df)

(println (java.util.Date.) n " --- " "before eval 7")
(->jvm (py. model eval_model eval-df))


;(println (java.util.Date.) n " --- " "before GC")
;(py/gc!)
;(println (java.util.Date.) n " --- " "end")

                                        ;        )
