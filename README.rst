funlib.evaluate
===============

.. image:: https://travis-ci.com/funkelab/funlib.evaluate.svg?branch=master
  :target: https://travis-ci.com/funkelab/funlib.evaluate

Popular metrics and reporting tools for volume evaluation. Currently includes
RAND, VOI, expected run length, and a metric to measure the number of splits
required to fix merges on a region adjacency graph. Also includes normalized 
VOI (NVI) and normalized information distance (NID) 
(https://dl.acm.org/doi/10.5555/1756006.1953024)

This module requires ``graph_tool`` to be installed. In a conda environment, get it via::

  conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
