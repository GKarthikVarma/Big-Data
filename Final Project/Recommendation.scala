import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.math.pow
import scala.io.Source
import java.io._

object collaboration {
  def main(args: Array[String])
  {
    val conf = new SparkConf().setAppName("Final Project")
    val sc = new SparkContext(conf)
    
    /*
     * Ratings
     */
    
    val ratings = sc.textFile(args(0))
    val first = ratings.first()
    var rating = ratings.filter(f=>f!=first).map{line =>
      val fields = line.split(",")
    Rating(fields(0).toInt,fields(1).toInt,fields(2).toDouble)}
    rating.take(2).foreach(println)
    
    /*
     * Movies
     */
    val movies = sc.textFile(args(1))
    val first_mov = movies.first()
    var movie = movies.filter(f=>f!=first_mov).map(line=>line.split(","))
      .map(a => (a(0).toInt,a(1).toString))
    //movie.collect().take(2).foreach(println)
    
    
    /*
     * Splitting the training data
    */
    val splits = rating.randomSplit(Array(0.8,0.2), seed = 0L)
    val training = splits(0).cache()
    val test = splits(1).cache().map{case Rating(user,product,rate) =>
      (user,product)}
    
    //training.collect().take(2).foreach(println)
    //test.collect().take(2).foreach(println)
    //validation.collect().take(2).foreach(println)
    
    /*
     * Training Phase
     * Set Parameters
     */
    val seed = 2L
    val numIterations = 10
    val learning_rate = 0.1
    val rank = 12
    
    /*
     * Training Phase
     * Train Model
     */
    val model = ALS.train(training, rank, numIterations,learning_rate)
    
    /*
     * Prediction
     */
    val predictions = model.predict(test).map{case Rating(user,product,rate) =>
      ((user,product),rate)}
    val ratesAndPreds = rating.map{case Rating(user,product,rate) =>
      ((user,product),rate)}.join(predictions)
      
    /*
     * MSE
     */
    val rmse = math.sqrt(ratesAndPreds.map{case ((user,product),(rating1,rating2))
      => val err = (rating1 - rating2)
      err * err
      }.mean())
      
    println("RMSE = " + rmse)
    //predictions.take(10).foreach(println)
    //ratesAndPreds.take(10).foreach(println)

    var final_predictions = predictions.map{case ((i,j),k)=>(j,(i,k))}.join(movie)
    //final_predictions.take(10).foreach(println)

    var output = final_predictions.map(f =>(f._2._1._1,(f._1,f._2._2,f._2._1._2)))
    //output.take(10).foreach(println)
    output.sortByKey(true).saveAsTextFile(args(2))
  }
}
