

import os
import argparse

import threading
from multiprocessing import Process
from multiprocessing import Manager

import time
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="highest_score_in_nbest.py [<args>] [-h | --help]"
    )

    parser.add_argument("--src", type=str, required=True,
                        help="Path of source file")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path of reference file")
    parser.add_argument("--hyp", type=str, required=True,
                        help="Path of trained models")
    parser.add_argument("--nbest", type=int, required=True,
                        help="hyp contain how many candidates?")

    # model and configuration
    parser.add_argument("--tmp_filename", type=str,required=True,
                        help="tmp file name for generate tmp file")
    parser.add_argument("--multi_bleu_script", type=str,required=True,
                        help="scripts for multi_bleu metric")
    parser.add_argument("--nThread", type=int,default=0,
                          help="hyp contain how many thread?")
    parser.add_argument("--njobs", type=int,default=0, help="how many process")

    return parser.parse_args()

class myThread(threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, obj, buck):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.obj = obj
        self.tmp = obj.tmp_filename + 'threadid{0}'.format(threadID)
        self.res = dict()
        self.buck = buck

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print("Starting " + self.name)
        self.res = self.obj(start=self.buck[0], end=self.buck[1], tmp=self.tmp)
        print("Exiting " + self.name)

    def get_res(self):
        return self.res


class ChooseHighestScoreInHyp():
    def __init__(self, src, ref, hyp, nbest, tmp_filename, multi_bleu_script):
        self.src = self.read_file(src)
        self.ref = self.read_file(ref)
        self.hyp = self.read_file(hyp)
        self.nbest = nbest if isinstance(nbest, int) else int(nbest)

        self.tmp_filename = tmp_filename
        self.multi_bleu_script = multi_bleu_script

        assert os.path.isfile(self.multi_bleu_script), self.multi_bleu_script

    def __call__(self, start, end, tmp, return_dict):
        res = self.highest_score_in_nbest(self.src, self.ref, self.hyp, self.nbest, start, end, tmp)
        return_dict[tmp] = res
        print("{0} ChooseHighestScoreInHyp has complete".format(tmp))
        return res

    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')
            while (lines !=[] and len(lines[-1])<1):
                lines.pop()

        return lines

    def eval_bleu(self, ref, nmt, tmp):
        with open(tmp + '.ref', 'w') as f:
            f.write(ref)
        with open(tmp + '.nmt', 'w') as f:
            f.write(nmt)
        ref_f = tmp + '.ref'
        nmt_f = tmp + '.nmt'
        score_f = tmp + ".score"
        os.system("perl {0} -lc {1} < {2} > {3}".format(self.multi_bleu_script, ref_f, nmt_f, score_f))
        os.system("{0} -lc {1} < {2} > {3}".format(self.multi_bleu_script, ref_f, nmt_f, score_f))

        bleu = self.read_file(tmp + ".score")

        os.remove(nmt_f)
        os.remove(ref_f)
        os.remove(score_f)

        if bleu == [] or bleu == '':
            return 0
        # BLEU = 72.74, 81.8/71.4/70.0/68.4 (BP=1.000, ratio=1.100, hyp_len=22, ref_len=20)
        bleu = bleu[0].split(',')[0].split('=')[1].strip()
        return float(bleu[0])

    def highest_score_in_nbest(self, src, ref, hyp, nbest, start, end, tmp):
        assert len(src) == len(ref),"length of src:{0} length of ref{1}".format(len(src), len(ref))
        assert len(src) * nbest == len(hyp),"length of src:{0} \t nbest{1} \t length of hyp:{2}".format(len(src), nbest, len(hyp))

        count = 0
        res = dict()
        for i in range(start, end):
            r = ref[i]
            if len(r) < 1:
                continue
            max_score = (0, -1)
            for j in range(nbest):
                if len(hyp[i * nbest + j]) < 1:
                    continue
                j_score = self.eval_bleu(r, hyp[i * nbest + j], tmp)
                if j_score > max_score[1]:
                    max_score = (i * nbest + j, j_score)
            if max_score[1] > 0:
                res[src[i]] = hyp[max_score[0]]

        return res


def main():
    params = parse_args()
    obj = ChooseHighestScoreInHyp(params.src, params.ref, params.hyp, params.nbest, params.tmp_filename, params.multi_bleu_script)
    jobs = []
    step = int(len(obj.ref) / params.njobs) if params.njobs else int(len(obj.ref) / params.nThread)
    bucket = [(i * step, i*step + step) for i in range(params.njobs)] if step > 0 else [(0, 0)]
    bucket[-1] = (bucket[-1][0], bucket[-1][1] + len(obj.ref) % params.njobs)
    manager = Manager()
    return_dict = manager.dict()
    for i in range(len(bucket)):
        buck = bucket[i]
        p = Process(target=obj, args=(buck[0], buck[1],params.tmp_filename +  "process%d"%i, return_dict))
        jobs.append(p)
        p.start()

    for i in jobs:
        i.join()

    with open(params.tmp_filename + '.src', 'w') as src_stream,open(params.tmp_filename + '.hyp.%d'%params.nbest, 'w') as tgt_stream:
        for res in return_dict.values():
            if res ==None or len(res) < 1:
                continue
            for key, value in res.items():
                src_stream.write(key + '\n')
                tgt_stream.write(value+ '\n')

if __name__ == "__main__":
    start = time.time()
    main()
    print("consuming time is ".format(time.time() - start))
