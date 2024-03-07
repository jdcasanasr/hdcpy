for i in {3..4}
do
    for j in {1000..10000..1000}
    do
        echo "D = $j | Q = $i"
        for k in {1..3}
        do
            rm -rf __pycache__
            python3 ucihar_unified_test.py "$j" "$i"
        done
    done
done