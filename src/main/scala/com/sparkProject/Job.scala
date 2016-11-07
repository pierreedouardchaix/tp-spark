package com.sparkProject

import org.apache.spark.sql.DataFrameReader
import org.apache.spark.sql.DataFrameWriter
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._


object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        Début du projet
      *
      ********************************************************************************/

    // Chargement des données
    val reader: DataFrameReader = spark.read
    val cumulative_reader = reader.option("sep",",").option("header",true).option("comment","#").option("inferSchema", "true")
    val cumulative_filepath = ???
    val cumulative = cumulative_reader.csv(cumulative_filepath)

    // Impression des 5 premières lignes
    cumulative.show(5)

    // Impression du nombre de lignes et de colonnes
    println("Nombre de lignes : " + cumulative.count())
    println("Nombre de colonnes : " + cumulative.columns.size)

    // Imprimer le dataset sous forme de table
    // cumulative.show()

    // Imprimer uniquement les colonnes 10 à 20
    val cols1020 = cumulative.columns.slice(9,19)
    cumulative.select(cols1020.head, cols1020.tail:_*).show()

    // Imprimer le schéma du data frame
    cumulative.printSchema()

    // Montrer le compte de koi_disposition par clé
    cumulative.select("koi_disposition").groupBy($"koi_disposition").count().show()

    /*
    CLEANING
     */

    // Conserver uniquement les lignes qui nous intéressent pour le modèle (koi_disposition = CONFIRMED ou FALSE POSITIVE )
    val indexOfkoi_disposition = cumulative.columns.indexOf("koi_disposition")
    val cumulative_cfp = cumulative.filter(row => row(indexOfkoi_disposition) != "CANDIDATE")

    // Afficher le nombre d’éléments distincts dans la colonne “koi_eccen_err1”.
    println(cumulative_cfp.select("koi_eccen_err1").groupBy($"koi_eccen_err1").count().count())

    // Enlever la colonne “koi_eccen_err1”.
    cumulative_cfp.drop("koi_eccen_err1")

    // La liste des colonnes à enlever du dataFrame avec une seule commande:
    /* La colonne “index" est en doublon avec “rowid”, on peut l’enlever sans perte d’information.
      "kepid" est l’Id des planètes.
      Les colonnes "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec” contiennent des informations provenant d’autres mesures que la courbe de luminosité. Si on garde ces colonnes le modèle est fiable dans 99.9% des cas ce qui est suspect ! On enlève donc ces colonnes.
      "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname", "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name", "koi_vet_date", "koi_pdisposition" ne sont pas essentielles.
    */

    val cumulative_cfp_drop = cumulative_cfp.drop("index","kepid","koi_eccen_err1", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname", "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name", "koi_vet_date", "koi_pdisposition")

    // D’autres colonnes ne contiennent qu’une seule valeur (null, une string, un entier, ou autre).
    // Elles ne permettent donc pas de distinguer les deux classes qui nous intéressent.
    // Trouver le moyen de lister toutes ces colonnes non-valides ou bien toutes les colonnes valides, et ne conserver que ces dernières.

    val col_analysis = cumulative_cfp_drop.columns.map{column_name: String => cumulative_cfp_drop.groupBy(column_name).count().count() != 1}

    val cols_to_keep = cumulative_cfp_drop.columns.zip(col_analysis).collect { case (x: String, true) => x }

    // On enlève les colonnes qui n'ont qu'une seule valeur.
    val cumulative_cfp_filtered = cumulative_cfp_drop.select(cols_to_keep.map(col):_*)

    // Stats sur les colonnes
    cumulative_cfp_filtered.describe("koi_disposition","koi_eccen").show()

    // Remplacer les valeurs null par 0
    val cumulative_cfp_final = cumulative_cfp_filtered.na.fill(0.0)

    // Joindre deux data frames
    println("Début de la jointure")
    val df_labels = cumulative_cfp_final.select("rowid", "koi_disposition")
    val df_features = cumulative_cfp_final.drop("koi_disposition")

    val cumulative_cfp_merge = df_features.join(df_labels, "rowid")

    // Ajouter et manipuler des colonnes

    println("Début calcul koi_ror_max et koi_ror_min")
    /*
    cumulative_cfp_merge.withColumn("koi_ror_max", cumulative_cfp_merge("koi_ror").cast("double") + cumulative_cfp_merge("koi_ror_err1").cast("double"))
      .withColumn("koi_ror_min", cumulative_cfp_merge("koi_ror").cast("double") + cumulative_cfp_merge("koi_ror_err2").cast("double"))
      .show(3)
    */
    // Avec une UDF

    val addColumns = {(a: Double, b: Double) => a + b}

    val addColumnsUDF = udf(addColumns)
    val cumulative_cfp_merge_output =
        cumulative_cfp_merge
          .withColumn("koi_ror_max", addColumnsUDF('koi_ror, 'koi_ror_err1))
          .withColumn("koi_ror_min", addColumnsUDF('koi_ror, 'koi_ror_err2))

    val cumulative_output_filePath = ???
    cumulative_cfp_merge_output.coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(cumulative_output_filePath)

  }


}
