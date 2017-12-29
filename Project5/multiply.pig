matrix1 = LOAD '$M' USING PigStorage(',') AS (row,column,value);
matrix2 = LOAD '$N' USING PigStorage(',') AS (row,column,value);

join1 = JOIN matrix1 BY column, matrix2 BY row;
multiply1 = FOREACH join1 GENERATE matrix1::row AS row,matrix2::column AS column,(matrix1::value)*(matrix2::value) AS value;
group1 = GROUP multiply1 BY (row, column);
add = FOREACH group1 GENERATE $0 as row, SUM(multiply1.value) AS value;


STORE add INTO '$O' USING PigStorage(',');
dump add
