library("getDEE2")

#packageVersion("getDEE2")
#sessionInfo()

# mdat<-getDEE2Metadata("hsapiens")

# # mdat1<-mdat[which((mdat$SRP_accession %in% c("SRP035268","SRP277202")) & 
                    # # (mdat$QC_summary %in% c("PASS","WARN(8)"))),]
# mdat1 <- mdat[which((mdat$SRP_accession %in% c("SRP035268","SRP277202")) & 
                    # (mdat$QC_summary == "PASS" | grepl("WARN", mdat$QC_summary))),]

# SRRlist<-as.vector(mdat1$SRR_accession)

# x <- getDEE2("hsapiens", SRRlist, metadata=mdat, counts = "GeneCounts",
             # outfile="T2D_DEE2_data.zip", legacy=TRUE)
			 
			 
#test <- getDEE2("hsapiens",SRRlist,counts = "GeneCounts",outfile="T2D_DEE2_test.zip",legacy = TRUE)
			 
			 
#x <- getDEE2("hsapiens", SRRlist, metadata=mdat, counts = "GeneCounts",outfile="T2D_DEE2_data.zip", legacy=FALSE)
			 
#x <- getDEE2("hsapiens",SRRlist,metadata = mdat,counts = "GeneCounts",outfile = "T2D_DEE2_data.zip")

#unzip("T2D_DEE2_data.zip", exdir = "dee2_data")
#geneCounts <- read.table("dee2_data/GeneCountMatrix.tsv",header = TRUE,row.names = 1,sep = "\t",check.names = FALSE)
#x <- list(GeneCounts = geneCounts)

#x$MetadataSummary

# Load metadata
mdat <- getDEE2Metadata("hsapiens")

# Studies of interest
target_srp <- c("SRP035268", "SRP277202")

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

    download.file(
        url,
        destfile = zipfile,
        mode = "wb"
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

# Combine all studies by gene ID
GeneCounts <- Reduce(
    function(x, y) merge(
        x,
        y,
        by = "row.names",
        all = TRUE
    ),
    count_list
)

dim(count_list[[1]])
dim(count_list[[2]])

head(rownames(count_list[[1]]))
head(rownames(count_list[[2]]))

rownames(GeneCounts) <- GeneCounts$Row.names
GeneCounts$Row.names <- NULL

# Missing genes in one study become 0
GeneCounts[is.na(GeneCounts)] <- 0

# Convert to matrix if downstream code expects a matrix
GeneCounts <- as.matrix(GeneCounts)

# Recreate getDEE2-style object
x <- list(
    GeneCounts = GeneCounts
)

##Adding the metadata of first dataset
sra.metadata=data.frame()
if(file.exists("SraRunTable_SRP035268.txt")){
  sra.metadata<-read.csv("SraRunTable_SRP035268.txt", header = TRUE)
}

#Adding the metadata of second dataset
if(file.exists("SraRunTable_SRP277202.txt")){
  sra.metadata2<-read.csv("SraRunTable_SRP277202.txt", header = TRUE)
}
sra.metadata[setdiff(names(sra.metadata2),names(sra.metadata))] <- NA
sra.metadata2[setdiff(names(sra.metadata),names(sra.metadata2))] <- NA

sra.metadata<-rbind(sra.metadata,sra.metadata2)

sra.metadata<-sra.metadata[which(sra.metadata$Run %in% mdat1$SRR_accession),]

sra.metadata$disease <- as.factor(as.numeric(grepl("palmitate",
                                sra.metadata$Treatment)))

write.csv(sra.metadata, file="T2D_metadata.csv", row.names=FALSE)

library("DESeq2")
x1<-x$GeneCounts[which(rowSums(x$GeneCounts)/ncol(x$GeneCounts)>=(10)),]
cat("CLASS x$GeneCounts:\n")
print(class(x$GeneCounts))

cat("DIM x$GeneCounts:\n")
print(dim(x$GeneCounts))

cat("CLASS x1:\n")
print(class(x1))

cat("DIM x1:\n")
print(dim(x1))

cat("HEAD COLNAMES:\n")
print(head(colnames(x1)))
x1<-x1[,which(colnames(x1) %in% sra.metadata$Run)]
dds <- DESeqDataSetFromMatrix(countData = x1, colData = sra.metadata, design = ~ disease)
res <- DESeq(dds)
z <- results(res)
vsd <- vst(dds, blind=FALSE)
zz <- cbind(z,assay(vsd))
zz <-as.data.frame(zz[order(zz$padj),])

zz$Ensembl <- rownames(zz)

library("biomaRt")
mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
annots <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id",
                "hgnc_symbol", "entrezgene_id", "entrezgene_accession", 
                "external_gene_name", "description"), 
                values=zz$Ensembl, mart= mart, useCache = FALSE)

#library(org.Hs.eg.db)
#ann_ent <- select(org.Hs.eg.db, keys=zz$Ensembl, columns=c("SYMBOL","ENTREZID"), 
#                        keytype="ENSEMBL")
#zz <- merge(zz, ann_ent, by.x=0, by.y="ENSEMBL")

zz <- merge(zz, annots, by.x="Ensembl", by.y="ensembl_gene_id")

#library("EnhancedVolcano")
#EnhancedVolcano(zz,lab=zz$hgnc_symbol,x="log2FoldChange",y="padj",
#                pCutoff = 10e-2, FCcutoff = 1,)

fold.cutoff=1
df<-zz[which(((zz$log2FoldChange>fold.cutoff) | 
                (zz$log2FoldChange<(fold.cutoff*-1))) & (zz$padj<0.05)),]

write.csv(df[,c("Ensembl","hgnc_symbol","entrezgene_id", "entrezgene_accession",
                "external_gene_name", "description","log2FoldChange","pvalue",
                "padj")], file="DEG_T2D_LFC1.csv", row.names=FALSE)

