"""Apply the same blinded semantic judge to anchor-writer outputs."""
import sys
import semantic_query_writer_rl as semantic
from qfad_anchor_writer import OUT, base

if __name__=='__main__':
    semantic.OUT=OUT
    semantic.EXTERNAL_CACHE_ROOTS=[base.OUT]
    semantic.main()
