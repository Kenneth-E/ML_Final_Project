
assess <- function(patient,
                   mean = 0,
                   sd = 1,
                   n = NULL,
                   controls = NULL,
                   CI = 95,
                   treshold = 0.05,
                   iter = 10000,
                   color_controls = "#2196F3",
                   color_CI = "#E91E63",
                   color_score = "black",
                   color_size = 2,
                   alpha_controls = 1,
                   alpha_CI = 0.8,
                   verbose = TRUE) {
  if (is.null(controls)) {
    if (is.null(n)) {
      if (verbose == TRUE) {
        warning("Sample size (n) not provided, thus set to 1000.")
      }
      n <- 1000
    }
  }




  # If score is list
  if (length(patient) > 1) {
    if (verbose == TRUE) {
      warning("Multiple scores were provided. Returning a list of results.")
    }
    results <- list()
    for (i in seq_len(length(patient))) {
      results[[i]] <- crawford.test(
        patient[i],
        controls,
        mean,
        sd,
        n,
        CI,
        treshold,
        iter,
        color_controls,
        color_CI,
        color_score,
        color_size,
        alpha_controls,
        alpha_CI
      )
      return(results)
    }
  } else {
    result <- crawford.test(
      patient,
      controls,
      mean,
      sd,
      n,
      CI,
      treshold,
      iter,
      color_controls,
      color_CI,
      color_score,
      color_size,
      alpha_controls,
      alpha_CI
    )
    return(result)
  }
}







