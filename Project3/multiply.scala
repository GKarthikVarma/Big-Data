
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

@SerialVersionUID(123L)
case class M_Matrix ( i: Long, j: Long, v: Double )
      extends Serializable {}

@SerialVersionUID(123L)
case class N_Matrix ( j: Long, k: Long, w: Double )
      extends Serializable {}

object Multiply {
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("Join")
    val sc = new SparkContext(conf)
    val M = sc.textFile(args(0)).map( line => { val a = line.split(",")
                                                M_Matrix(a(0).toLong,a(1).toLong,a(2).toDouble)})
    val N = sc.textFile(args(1)).map( line => { val a = line.split(",")
                                                N_Matrix(a(0).toLong,a(1).toLong,a(2).toDouble)})
    
    val M_Indexj = M.map(M => (M.j,(M.i,M.v)))
    val N_Indexj = N.map(N => (N.j,(N.k,N.w)))
    val MN_Join = M_Indexj.join(N_Indexj)
    
    val reduce = MN_Join.map({case (j,((i,v),(k,w))) => ((i,k),v*w)}).reduceByKey(_ + _)
    //val final_MN = reduce.map({case ((i,k),v) => (i,k,v)})
    reduce.sortByKey(true).collect().foreach(println)
    reduce.sortByKey(true).saveAsTextFile(args(2))
    sc.stop()
    
    
    }
  }
