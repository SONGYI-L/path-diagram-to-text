import codecs
from argparse import ArgumentParser
from tempfile import mkdtemp
import os
import shutil
import subprocess
import re
import sys
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def read_lines(file_name, multi_ref=False):
    """Read one instance per line from a text file. In multi-ref mode, assumes multiple lines
    (references) per instance & instances separated by empty lines."""
    buf = [[]] if multi_ref else []
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            line = line.strip()
            if multi_ref:
                if not line:
                    buf.append([])
                else:
                    buf[-1].append(line)
            else:
                buf.append(line)
    if multi_ref and not buf[-1]:
        del buf[-1]
    return buf

def read_json(json_file):
    """Read the JSON file and extract the 'text' field."""
    with codecs.open(json_file, 'r', 'UTF-8') as fh:
        data = json.load(fh)
    texts = [item['text'] for item in data]
    return texts

def read_tsv(tsv_file):
    """Read a TSV file, check basic integrity."""
    tsv_data = read_lines(tsv_file)
    tsv_data[0] = re.sub(u'\ufeff', '', tsv_data[0])  # remove unicode BOM
    tsv_data = [line.split("\t") for line in tsv_data if line]  # split, ignore empty lines

    # remove quotes
    refs = []
    for _, ref in tsv_data:
        ref = re.sub(r'^\s*[\'"]?\s*', r'', ref)
        ref = re.sub(r'\s*[\'"]?\s*$', r'', ref)
        refs.append(ref)
    # check quotes
    errs = [line_no for line_no, sys in enumerate(refs, start=1) if '"' in sys]
    if errs:
        print("%s -- has quotes" % tsv_file)
        raise ValueError('%s -- Quotes on lines: %s' % (tsv_file, str(errs)))

    return refs


def load_data(ref_file, sys_file):
    """Load the data from the given files."""
    data_ref = read_lines(ref_file, multi_ref=True)
    data_sys = read_json(sys_file)

    # sanity check
    assert len(data_ref) == len(data_sys), "{} != {}".format(len(data_ref), len(data_sys))
    return data_ref, data_sys

def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out

def evaluate(data_ref, data_sys, print_as_table=False, print_table_header=False, sys_fname=''):
    """Main procedure, running the MS-COCO evaluators on the loaded data."""

    # run the MS-COCO evaluator
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}

    metric_names = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    if print_as_table:
        if print_table_header:
            print('\t'.join(['File'] + metric_names))
        print('\t'.join([sys_fname] + ['%.4f' % scores[metric] for metric in metric_names]))
    else:
        print('SCORES:\n==============')
        for metric in metric_names:
            print('%s: %.4f' % (metric, scores[metric]))


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def main():
    parser = ArgumentParser()
    parser.add_argument('ref_file', help='JSON file containing the reference captions')
    parser.add_argument('sys_file', help='JSON file containing the system output captions')
    parser.add_argument('-t', '--print_as_table', action='store_true',
                        help='print scores as a table (default: print as individual scores)')
    parser.add_argument('-H', '--table_header', action='store_true',
                        help='include a header in table output (ignored if -t is not set)')
    args = parser.parse_args()

    ref_file = args.ref_file
    sys_file = args.sys_file

    # load data
    data_ref, data_sys = load_data(ref_file, sys_file)

    # evaluate
    evaluate(data_ref, data_sys, print_as_table=args.print_as_table, print_table_header=args.table_header,
             sys_fname=sys_file)


if __name__ == '__main__':
    main()
