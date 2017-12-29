drop table Matrix1;
drop table Matrix2;
drop table MultipliedMatrix;

create table Matrix1(
  i int,
  j int,
  value double)
row format delimited fields terminated by ',' stored as textfile;

create table Matrix2(
i int,
j int,
value double)
row format delimited fields terminated by ',' stored as textfile;

create table MultipliedMatrix(
i int,
j int,
value double
)
row format delimited fields terminated by ',' stored as textfile;

load data local inpath '${hiveconf:M}' overwrite into table Matrix1;

load data local inpath '${hiveconf:N}' overwrite into table Matrix2;

INSERT INTO TABLE MultipliedMatrix
select m1.i,m2.j,sum(m1.value*m2.value) 
from Matrix1 as m1 join Matrix2 as m2 on m1.j=m2.i
GROUP BY m1.i,m2.j;

Select count(i),avg(value) from MultipliedMatrix;

