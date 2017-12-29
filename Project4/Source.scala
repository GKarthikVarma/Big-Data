import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import java.lang.Long

@SerialVersionUID(123L)
case class Edges ( i: Int, d: Int, j: Int )
      extends Serializable {}

  object Source {

    def min(a: Int,b:Int): Int={
      if(a>b)
      {
        return b
      }else{return a}
    }
    
    def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("Shortest Distance")
      // conf.setMaster("local[2]")
      val sc = new SparkContext(conf)
      //var input = sc.textFile(args(0)).map(line => {val a = line.split(",")
      //  (a(0).toInt, a(1).toInt, a(2).toInt)})
      
    
      val input = sc.textFile(args(0)).map( line => { val a = line.split(",")
                                               Edges(a(0).toInt,a(1).toInt,a(2).toInt)})
     

      var pre_dist = input.groupBy(_.j)
      var tuple = input.map(e => (e.i, 750000, e.d, e.j))
      var distance = pre_dist.map(e => if (e._1 == 0) 
        {(e._1, 0)} 
        else {(e._1, 750000)})
      for (i <- 1 to 4) {    
        tuple = tuple.map(e => (e._4, e)).join(distance.map(f => (f._1, f)))
        .map ({ case (a, (b, c)) => (b._1, c._2.toString.toInt, b._3, b._4) })

        distance = tuple.map(e => (e._1, e)).join(distance.map(f => (f._1, f)))
        .map ({ case (a, (b,c)) => (b._4, min(c._2 + b._3, b._2))})
        .reduceByKey(_ min _)
      }
      distance = distance.sortBy(_._1)
      distance = distance.filter(e => (e._2 != 750000))
      //distance = distance.map(e => (e._1,e._2+Int.MaxValue))
      distance.collect.foreach(println)
      distance.saveAsTextFile(args(1))
    }

  }
