library("getDEE2")
library(jsonlite)
library(XML)
library("DESeq2")

options(timeout = 1500)

# args <- commandArgs(trailingOnly = TRUE)

# if (length(args) == 0) {
#   stop("Please provide at least one SRP accession")
# }

# target_srp <- args

# print(target_srp)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript script.R <control_name> <test_name> <SRP1> [SRP2 ...]")
}

control_name <- args[2]
test_name <- args[1]
log2fold <- as.numeric(args[3])
pval <- as.numeric(args[4])
target_srp <- args[-c(1, 2, 3, 4)]

cat("Control:", control_name, "\n")
cat("Test:", test_name, "\n")
cat("Log2FoldChange:", log2fold, "\n")
cat("Padj:", pval, "\n")
cat("SRP IDs:\n")
print(target_srp)

retry_download <- function(expr) {
  for (i in 1:2) {
    result <- try(eval.parent(substitute(expr)), silent = TRUE)
    if (!inherits(result, "try-error")) {
      return(result)
    }

    if (i == 1) {
      message("First attempt failed. Retrying...")
      Sys.sleep(2)
    }
  }

  stop(result)
}

retry_void <- function(expr) {
  for (i in 1:2) {
    result <- try(eval.parent(substitute(expr)), silent = TRUE)
    if (!inherits(result, "try-error")) {
      return(invisible(NULL))
    }

    if (i == 1) {
      message("First attempt failed. Retrying...")
      Sys.sleep(2)
    }
  }

  stop(result)
}

cache_dir <- "cache"

if (!dir.exists(cache_dir)) {
  dir.create(cache_dir, recursive = TRUE)
}

get_cached_dee2_metadata <- function(species = "hsapiens") {

  cache_file <- file.path(
    cache_dir,
    paste0(species, "_metadata.rds")
  )

  if (file.exists(cache_file)) {

    message("Loading DEE2 metadata from cache...")
    return(readRDS(cache_file))

  } else {

    message("Downloading DEE2 metadata...")
    mdat <- getDEE2Metadata(species)

    saveRDS(mdat, cache_file)

    return(mdat)
  }
}

mdat <- get_cached_dee2_metadata("hsapiens")

# Filter metadata
mdat1 <- subset(
    mdat,
    SRP_accession %in% target_srp &
    QC_summary %in% c("PASS", "WARN(8)")
)

# Unique studies (SRP-GSE pairs)
studies <- unique(
    mdat1[, c("SRP_accession", "GEO_series")]
)

count_list <- list()

print(studies)
cat("Number of studies:", nrow(studies), "\n")

for(i in seq_len(nrow(studies))) {

    srp <- studies$SRP_accession[i]
    gse <- studies$GEO_series[i]

    # Build URL dynamically
    zipfile <- sprintf("%s_%s.zip", srp, gse)

    url <- sprintf(
        "https://dee2.io/huge/hsapiens/%s",
        zipfile
    )

    message("Downloading: ", zipfile)

    # download.file(
    #     url,
    #     destfile = zipfile,
    #     mode = "wb"
    # )
    retry_void(
        download.file(
        url,
        destfile = zipfile,
        mode = "wb"
      )
)

    # Extract to study-specific folder
    exdir <- sprintf("%s_%s", srp, gse)

	# Unzip
	unzip(zipfile, exdir = exdir)

	# Find GeneCountMatrix.tsv anywhere under the extracted folder
	count_file <- list.files(
		exdir,
		pattern = "^GeneCountMatrix\\.tsv$",
		recursive = TRUE,
		full.names = TRUE
	)
	

	if(length(count_file) != 1) {
		stop(
			sprintf(
				"Expected 1 GeneCountMatrix.tsv in %s, found %d",
				zipfile,
				length(count_file)
			)
		)
	}

	counts <- read.delim(
		count_file,
		row.names = 1,
		check.names = FALSE
	)
	
	count_list[[paste0(srp, "_", gse)]] <- counts
	
	cat("\n")
	cat("Study:", srp, gse, "\n")
	print(dim(counts))
	print(head(colnames(counts)))
}

if (length(count_list) == 1) {

    GeneCounts <- count_list[[1]]

} else {

    GeneCounts <- Reduce(
        function(x, y) merge(
            x,
            y,
            by = "row.names",
            all = TRUE
        ),
        count_list
    )

    rownames(GeneCounts) <- GeneCounts$Row.names
    GeneCounts$Row.names <- NULL
}

GeneCounts[is.na(GeneCounts)] <- 0
GeneCounts <- as.matrix(GeneCounts)

# Recreate getDEE2-style object
x <- list(
    GeneCounts = GeneCounts
)

# Function to download RunInfo metadata for an SRP
get_sra_metadata <- function(srp) {

  runinfo_url <- paste0(
    "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=",
    srp
  )

  # read.csv(runinfo_url)
  retry_download(read.csv(runinfo_url))
}

# Function to retrieve treatment from BioSample
get_treatment_from_biosample <- function(biosample_acc) {

  search_url <- paste0(
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?",
    "db=biosample&term=", biosample_acc, "[accn]",
    "&retmode=json"
  )

  # search <- jsonlite::fromJSON(search_url)
  search <- retry_download(jsonlite::fromJSON(search_url))

  if (length(search$esearchresult$idlist) == 0)
    return(NA_character_)

  uid <- search$esearchresult$idlist[1]

  fetch_url <- paste0(
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?",
    "db=biosample&id=", uid,
    "&retmode=xml"
  )

  # xml_txt <- paste(
  #   readLines(fetch_url, warn = FALSE),
  #   collapse = "\n"
  # )
  xml_txt <- paste(
      retry_download(
      readLines(fetch_url, warn = FALSE)
    ),
    collapse = "\n"
  ) 

  doc <- xmlParse(xml_txt, asText = TRUE)

  attrs <- xpathApply(doc, "//Attribute", function(x) {
    c(
      name = xmlGetAttr(x, "attribute_name"),
      value = xmlValue(x)
    )
  })

  attrs <- do.call(rbind, attrs)

  idx <- which(tolower(attrs[, "name"]) == "treatment")
  # idx <- which(tolower(attrs[, "name"]) == "source_name")
  # idx <- which(
  #   tolower(attrs[, "name"]) == "source_name" |
  #   tolower(attrs[, "name"]) == "treatment"
  # )
  cat("ID:", idx)

  if (length(idx) == 0)
    return(NA_character_)

  attrs[idx[1], "value"]
}

# Download metadata for all studies
target_srp <- unique(mdat1$SRP_accession)

metadata_list <- lapply(target_srp, get_sra_metadata)

# Combine metadata
sra.metadata <- do.call(rbind, metadata_list)

# Keep only SRRs used in the DEE2 analysis
sra.metadata <- sra.metadata[
  sra.metadata$Run %in% mdat1$SRR_accession,
]

# Query each unique BioSample once
unique_bs <- unique(sra.metadata$BioSample)

treatment_map <- setNames(
  sapply(unique_bs, get_treatment_from_biosample),
  unique_bs
)

# Add treatment column
sra.metadata$treatment <- treatment_map[
  sra.metadata$BioSample
]

# # Original disease assignment
# sra.metadata$disease <- as.factor(
#   as.numeric(
#     grepl(
#       "COPD lung tissue",
#       sra.metadata$treatment,
#       ignore.case = TRUE
#     )
#   )
# )

sra.metadata <- sra.metadata[
    sra.metadata$treatment %in% c(control_name, test_name),
]

sra.metadata$disease <- factor(
    sra.metadata$treatment,
    levels = c(test_name, control_name)
)

# sra.metadata <- sra.metadata[
#     sra.metadata$treatment %in% c("Untreated", "Dexamethasone"),
# ]

# sra.metadata$disease <- factor(
#     sra.metadata$treatment,
#     levels = c("Untreated", "Dexamethasone")
# )

write.csv(
  sra.metadata,
  file = "T2D_metadata.csv",
  row.names = FALSE
)

x1<-x$GeneCounts[which(rowSums(x$GeneCounts)/ncol(x$GeneCounts)>=(10)),]
x1<-x1[,which(colnames(x1) %in% sra.metadata$Run)]
dds <- DESeqDataSetFromMatrix(countData = x1, colData = sra.metadata, design = ~ disease)
res <- DESeq(dds)
z <- results(res)
vsd <- vst(dds, blind=FALSE)
zz <- cbind(z,assay(vsd))
zz <-as.data.frame(zz[order(zz$padj),])

zz$Ensembl <- rownames(zz)

library("biomaRt")
mart <- useEnsembl(
    biomart = "genes",
    dataset = "hsapiens_gene_ensembl",
    mirror = "www"
)
annots <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id",
                "hgnc_symbol", "entrezgene_id", "entrezgene_accession", 
                "external_gene_name", "description"), 
                values=zz$Ensembl, mart= mart, useCache = FALSE)

zz <- merge(zz, annots, by.x="Ensembl", by.y="ensembl_gene_id")

fold.cutoff=log2fold
df<-zz[which(((zz$log2FoldChange>fold.cutoff) | 
                (zz$log2FoldChange<(fold.cutoff*-1))) & (zz$padj<pval)),]

write.csv(df[,c("Ensembl","hgnc_symbol","entrezgene_id", "entrezgene_accession",
                "external_gene_name", "description","log2FoldChange","pvalue",
                "padj")], file="DEG_T2D_LFC1.csv", row.names=FALSE)

