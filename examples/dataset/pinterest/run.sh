if [ ! -f "pinterest.edges" ]; then
    gunzip -k pinterest.edges.gz
fi

if [ ! -f "pinterest.test" ]; then
    gunzip -k pinterest.test.gz
fi

python ../../../scripts/learn.py pinterest.edges none cloverleaf --embedding-propagation 10 10 --dims 128 --valid-pct 0 --batch-size 1024 --neighborhood-alignment 0.3

python evaluate.py cloverleaf.node-embeddings pinterest.test
