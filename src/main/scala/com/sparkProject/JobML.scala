package com.sparkProject

import org.apache.spark.sql.DataFrameReader
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.SparseVector
import java.io.File
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


/**
  * Created by chaixpe on 27/10/2016.
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // Récupération des arguments de la ligne de commande :
    //          argsParsed(0) est le type d'input file,
    //          argsParsed(1) le chemin vers l'input file,
    //          argsParsed(2) le chemin vers le fichier d'hyperparamètres ou "" si l'utilisateur ne l'a pas spécifié (cet argument étant optionnel)
    val argsParsed = args match {
      case (Array("csv" | "parquet", _)) => Array(args(0), args(1), "") // Cas où l'utilisateur n'a pas donné de fichier d'hyperparamètres en argument
      case (Array("csv" | "parquet", _, _)) => args // Cas où l'utilisateur a donné un fichier d'hyperparamètres en argument
      case (_) => throw new IllegalArgumentException(usage); // Cas où l'utilisateur n'a pas spécifié "csv" ou "parquet" en premier argument, ou a donné plus de 3 arguments
    }

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech 4_5")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN") // Pour afficher moins en console
    import spark.implicits._

    // Import du fichier créé dans le TP précédent (csv ou parquet)
    val cumulative = argsParsed(0) match {
      case("csv") => {
        spark.read.option("sep",",").option("header",true).option("comment","#").option("inferSchema", "true").csv(args(1))
      }
      case("parquet") => {
        spark.read.parquet(args(1))
      }
    }

    // UDF pour créer la colonne de label
    val dummy = udf({ (l: String) => {
      if (l == "CONFIRMED") 0.0
      else if (l == "FALSE POSITIVE") 1.0
      else 0.0 // En réalité, il n'y a que CONFIRMED et FALSE POSITIVE dans le dataset.
    }
    })

    // Application au dataset et suppression des colonnes rowid et koi_disposition
    val cumulative_forML = cumulative.withColumn("label", dummy($"koi_disposition")).drop("rowid", "koi_disposition")

    // Colonnes à considérer pour le vecAssembler
    val vecAssemblerCols = cumulative_forML.columns.filter({s: String => if(s == "label") false else true})

    // Construction du vectorassembler (chaque vecteur est de type org.apache.spark.ml.linalg.Vector : SparseVector ou DenseVector)
    val assembler = new VectorAssembler()
      .setInputCols(vecAssemblerCols)
      .setOutputCol("featuresSparse")

    // Transformation
    val cumulative_forML_vecassembler = assembler.transform(cumulative_forML)

    // L'output de VectorAssembler est aléatoirement de type SparseVectors ou de type DenseVectors (sous-classes de Vector).
    // Pour StandardScaler, l'input doit obligatoirement être un DenseVector.
    // Transformation des vecteurs en Densevector via une UDF
    val sparseToDense = udf({ l: org.apache.spark.ml.linalg.Vector => {
       val composants = l.toArray
       new DenseVector(composants)
    }
    })

    val finalData = cumulative_forML_vecassembler.select("label", "featuresSparse").withColumn("features", sparseToDense($"featuresSparse")).drop("featuresSparse")

    // Régression avec ML

    // Centrer-réduire les données
    print("Fit d'un standard scaler\n")
    val scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true)
                .fit(finalData)

    val finalDataCR = scaler.transform(finalData).drop("features")

    // Split du dataset entre entraînement et test
    val finalDataSplit = finalDataCR.randomSplit(Array(0.9, 0.1), seed = 12345)
    val training = finalDataSplit(0)
    val test = finalDataSplit(1)

    // Initialisation de la régression logistique
    println("Début régression")
    val lr = new LogisticRegression()
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setStandardization(false)  // we already scaled the data
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setMaxIter(300)  // a security stop criterion to avoid infinite loops

    // Grille de paramètres à tester
    println("Début construction de grille de regParam")

    // Valeur d'hyperparamètres par défaut
    val regParamArrayDefault =  Array(-6.0, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, -0.0)
    val regParamArrayExpDefault = regParamArrayDefault.map({ x: Double => math.pow(10.0, x) })

    // Construction de la grille à partir des hyperparamètres (par défaut ou fournis par l'utilisateur)
    // On teste si l'utilisateur a passé en ligne de commande un fichier vers une liste d'hyperparamètres à tester.
    // Si ce n'est pas le cas ou bien s'il y a un problème pour lire les hyperparamètres du fichier, les valeurs par défaut sont utilisées.
    val regParamArrayExp: Array[Double] = argsParsed(2) match {
      case "" => println(nohyperParameterFile); regParamArrayExpDefault;
      case _ => {
        val hyperparametersFilePath: String = argsParsed(2)
        val source = scala.io.Source.fromFile(hyperparametersFilePath)
        val lines = try source.mkString.split("\n") finally source.close()
        lines.length match {
          case 1 => {
            try {
              val regParamArrayExpUser: Array[Double] = lines(0).split(";").map({s: String => s.toDouble})
              regParamArrayExpUser.length match {
                case 0 => println(notEnoughParameters); regParamArrayExpDefault;
                case _ => println(hyperParametersOK); regParamArrayExpUser;
              }
            }
            catch {
              case nfe: java.lang.NumberFormatException => println(hyperParametersNotOK); regParamArrayExpDefault;
            }
          }
          case _ => println(tooManyLines); regParamArrayExpDefault;
        }
      }
    }

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regParamArrayExp)
      .build()

    // Initialisation de la train validation
    println("Début train validation")
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setSeed(123456)

    // Fit du modèle
    println("Début fit")
    val lrModel = trainValidationSplit.fit(training)

    // Transformation du test set
    println("Début transformation du dataset test")
    val test_withpredictions = lrModel.transform(test)

    // Récupération des informations concernant le meilleur modèle
    val bestModel = lrModel.bestModel.asInstanceOf[LogisticRegressionModel]
    print("Intercept : ")
    println(bestModel.intercept)
    print("Coefficients : ")
    println(bestModel.coefficients)
    println("Meilleur paramètre de régression logistique : 10^%.1f".format(math.log10(bestModel.getRegParam)))
    //println(bestModel.getRegParam)

    // Précision du modèle retenu
    println("***\nRésultats du modèle sur le test set")
    test_withpredictions.groupBy("label", "prediction").count.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(test_withpredictions)
    println("Précision du modèle : %.2f".format(accuracy*100.0) + "%")

    // Enregistrement du modèle (avec remplacement s'il existe déjà) puis chargement à nouveau
    val savepath = "target/tmp/scalaLinearRegressionWithSGDModel"
    FileUtils.forceDelete(new File(savepath))

    bestModel.save(savepath)
    val sameModel = LogisticRegressionModel.load("target/tmp/scalaLinearRegressionWithSGDModel")
  }

  // Messages console lors de la lecture des arguments (fichier input et fichier d'hyperparamètres)

  val usage =
    """
      Arguments à fournir : [type de fichier d'input : csv ou parquet] [lien vers fichier input] [lien vers fichier d'hyperparamètres (optionnel)]
    """

  val nohyperParameterFile =
    """
      Aucun fichier d'hyperparamètre à tester n'a été donné en argument. Les valeurs par défaut seront utilisées.
      Ajouter le chemin vers un fichier d'hyperparamètres (sur une ligne, séparateur décimal ".", séparateur de champ ";") en argument de la commande pour les prendre en compte.
    """

  val tooManyLines =
    """
      Attention : le fichier d'hyperparamètres contient plusieurs lignes. Réessayer avec un fichier contenant une seule ligne.
      Le fichier a été ignoré et les valeurs par défaut seront utilisées.
    """

  val notEnoughParameters =
    """
      Attention : il n'y pas assez de paramètres à tester dans le fichier. Réessayer avec un fichier contenant au moins 1 paramètre (ou plusieurs paramètres séparés par ';').
      Le fichier a été ignoré et les valeurs par défaut seront utilisées.
    """

  val hyperParametersOK =
    """
        Lecture des hyperparamètres fournis par l'utilisateur : OK.
    """

  val hyperParametersNotOK =
    """
        Problème pendant la lecture des hyperparamètres. Le fichier a été ignoré et les valeurs par défaut seront utilisées.
    """


}
