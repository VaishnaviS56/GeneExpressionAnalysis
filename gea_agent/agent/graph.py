from __future__ import annotations

from typing import cast

import networkx as nx
from langgraph.graph import END, START, StateGraph

from gea_agent.agent.state import AgentState, Route
from gea_agent.config import SETTINGS
from gea_agent.tools.classify_query import classify_query
from gea_agent.tools.enrichr import enrichr_pathways
from gea_agent.tools.llm import get_llm
from gea_agent.tools.random_walk_restart import top_rwr_genes
from gea_agent.tools.string_local_graph import build_weighted_graph_from_string_files
from gea_agent.tools.synthesizer import synthesize_technical_response


def _route(state: AgentState) -> Route:
    classification = state.get("classification")
    if not classification:
        return "general"
    return cast(Route, "technical" if classification["kind"] == "technical" else "general")


def node_classify(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    classification = classify_query(query)
    return {"classification": classification, "genes": classification.get("genes", [])}


def node_general_answer(state: AgentState) -> AgentState:
    query = state.get("query")
    llm = get_llm()
    resp = llm.invoke(
        [
            ("system", "You are a helpful assistant. Answer the user clearly and concisely."),
            ("user", query),
        ]
    )
    return {"answer": getattr(resp, "content", "")}


def node_fetch_string(state: AgentState) -> AgentState:
    """Build STRING graph from downloaded local files."""
    genes = state.get("genes")
    print("genes=",genes)
    #genes = ["A1", "A1A", "A1B", "A2", "A2A", "A2B", "A3", "A431", "A549", "AAV5", "ABCA1", "ABCG1", "ACE2", "ACF1", "ACTION7", "AD4BP", "AD5", "ADAM9", "ADAR1", "ADAR2", "ADAR3", "ADF1", "AF2", "AGO1", "AGO2", "AID2", "AIM2", "AK1", "AKT1", "AKT2", "ALA374", "ALK4", "ALK7", "ALKBH5", "ALPHA1", "ALPHA1G", "ALPHA1H", "ALPHA1I", "ALPHA5", "ALR2", "AMHR2", "AML1", "ANKRD30A", "ANTIGEN1", "AOF2", "AP1", "AP2", "AP3", "APC1638N", "APETALA3", "ARF6", "ARG306", "ARP2", "ASIC1A", "ASK1", "ASN341", "ASP302", "AT3", "ATDMC1", "ATF1", "ATF6", "ATG1", "ATG11", "ATG32", "ATG7", "ATG8", "ATG8A", "ATREC8", "ATSPO11", "ATT20", "B1", "B15", "B19", "B1A", "B1B", "B2", "B23", "B2A", "B2B", "B4", "B5", "B558", "BALANCE11", "BCL2", "BD1", "BETA0", "BETA1", "BETA4", "BH3", "BHC110", "BICD2", "BLT1", "BLT2", "BMP15", "BMP16", "BMP2", "BMP4", "BMP7", "BMPR1B", "BNIP3", "BNIP3L", "BP1", "BPAG1", "BPAG2", "BQ123", "BQ3020", "BRAIN8", "BRCA1", "BRCA2", "BRG1", "BRI1", "BSP1", "BTG4", "BUBR1", "BWSIC1", "BWSIC2", "BY2", "BZLF1", "C13", "C16", "C17", "C19", "C2", "C20", "C23", "C3", "C381", "C4", "C447", "C57BL", "C5B", "C6", "CA1", "CA125", "CA2", "CA3", "CA4", "CAC1", "CAC2", "CAC3", "CACNA1S", "CACO2", "CAF1", "CAK1", "CAS9", "CAT1", "CB1", "CB2", "CBLL1", "CC2", "CC3", "CC50", "CCK8", "CCL4", "CCMP1779", "CCND1", "CCND2", "CCR1", "CCR3", "CCR4", "CCR5", "CCR7", "CCT5", "CCT8", "CD1", "CD105", "CD11B", "CD14", "CD147", "CD151", "CD2", "CD22", "CD25", "CD29", "CD34", "CD36", "CD38", "CD4", "CD40", "CD41A", "CD44", "CD47", "CD52", "CD59", "CD63", "CD73", "CD8", "CD87", "CD90", "CD95", "CD95L", "CD98", "CDC13", "CDC2", "CDC20", "CDC25", "CDC25C", "CDC27", "CDC28", "CDC4", "CDC42", "CDC42HS", "CDC45", "CDC48", "CDC5", "CDH1", "CDK1", "CDK2", "CDK4", "CDK5", "CDK6", "CDK7", "CDKN1C", "CELLS1", "CEMKLP1", "CEP55", "CETN1", "CF1", "CF2", "CFP1", "CG1652", "CG1656", "CG9997", "CGPX4", "CHANNELS10", "CHANNELS2", "CHK1", "CHK2", "CHO1", "CHX17", "CHX25", "CIG1", "CIRCHIPK3", "CIRCNRIP1", "CIRCRHOT1", "CIV1", "CK1", "CK2", "CKS1", "CKS2", "CKSHS1", "CKSHS2", "CLB1", "CLB2", "CLB3", "CLB4", "CLN1", "CNS4", "CO1", "CO2", "COPT1", "COPT2", "COPT3", "COPT4", "COPT5", "COPT6", "COPT7", "COQ10", "COQ6", "COR1", "CORTBP1", "COV2", "CP1", "CP110", "CP190", "CP60", "CPEB1", "CPLA2", "CPLX1", "CPP32", "CPSF6", "CR3", "CRISPLD1", "CRISPLD2", "CRM1", "CSE4", "CSE4P", "CSF1", "CSF1R", "CTLA4", "CUL3", "CUT5", "CXCL10", "CXCL11", "CXCL12", "CXCL9", "CXCR3", "CXCR4", "CYCA1", "CYP1", "CYP11A", "CYP17", "CYP19", "CYP1A", "CYP2", "CYP26", "CYP2D", "CYP3", "CYP79S", "D1", "D2", "D28", "D3", "D4E", "D4T", "D538", "DAX1", "DCAS9", "DCL1", "DCL4", "DCR2", "DDX4", "DE2F", "DELTA12", "DELTA4", "DELTA5", "DELTA9", "DELTABD1", "DELTANP73", "DFMR1", "DGRIP84", "DGRIP91", "DICER1", "DIO1", "DIO2", "DISEASE7", "DLK1", "DM2", "DMC1", "DMR1", "DMRT1", "DMRT1BY", "DN3", "DNAH1", "DNMT1", "DNMT2", "DNMT3", "DNMT3A", "DNMT3B", "DNMT3L", "DPC4", "DPPA5", "DPR2", "DPRMT5", "DR1", "DR2", "DRP1", "DS10", "DS28", "DTDC1", "DTDC2", "DUOX1", "DUOX2", "DUX4", "DYNAMICS8", "E0", "E1", "E12", "E13", "E1370", "E14", "E16", "E1A", "E1B", "E1S", "E2", "E2F", "E2S", "E3", "E3S", "E4", "E47", "E523", "E6", "E7", "E9", "EAAT1", "EAAT2", "EAAT3", "EAG1", "EC50", "ECAT1", "ECC1", "ECV304", "ED50", "EDG2", "EDG4", "EDG7", "EEA1", "EG5", "EGFL7", "EIF2", "EIF3B", "EIF4A", "EIF4E", "EIF4G", "EKHIDNA2", "ELK2", "ELK3", "ELOVL1", "ELSPBP1", "EM5", "EMBRYO1", "EMBRYOS1", "EMI1", "EMI2", "EP2", "ERALPHA1", "ERG1", "ERG2", "ERG3", "ERK1", "ERK2", "ERK5", "ERP72", "ESA1", "ESR1", "ETS1", "EXON2", "EXP5", "EXPORTIN5", "EZH2", "F0", "F1", "F1534C", "F2", "F3", "F4", "F74G", "F9", "FAB1", "FABP12", "FACTORS2", "FACTORS5", "FACTORS6", "FADH2", "FAM111A", "FB1", "FDZ1", "FDZ2", "FE2", "FGF23", "FGF4", "FGFR2", "FGFR3", "FIP3", "FK506", "FKB51", "FKBP5", "FKBP51", "FLOW4", "FLT3", "FMN2", "FMR1", "FOXD3", "FOXH1", "FOXL2", "FOXM1", "FOXO3", "FOXO3A", "FOXP3", "FRIZZLED3", "FRIZZLED6", "FUNDC1", "FUSION1", "FUT2", "FXR1", "FXR2", "FZ3", "FZ6", "G0", "G1", "G169", "G2", "G21", "G4", "G418", "G551D", "G9A", "GABRA2", "GAD1", "GAD65", "GAD67", "GADD34", "GADD45A", "GAL1", "GAL4", "GAPD2", "GAPR1", "GATA4", "GATA6", "GBA1", "GBA2", "GCN2", "GCN4", "GCN5", "GD3", "GDF15", "GDF3", "GDF8", "GDF9", "GENETICS7", "GIRK1", "GIRK4", "GJA4", "GLAND1", "GLI1", "GLIPR1", "GLO1", "GLU373", "GLUT2", "GLUT4", "GNRH2", "GNRH3", "GP120", "GP130", "GPR30", "GPX1", "GPX4", "GR1", "GR110", "GR66A", "GRCH37", "GRP75", "GRP78", "GRP94", "GRSF1", "GSK3", "GT11", "GTL2", "GW182", "GW4064", "H1", "H19", "H1K", "H1T", "H2A", "H2AX", "H2B", "H2S", "H3", "H4", "H524", "HABITATS1", "HADAR3", "HAND2", "HAP1", "HAS2", "HAS3", "HBA1C", "HCAS9", "HCD2", "HDAC1", "HDAC10", "HDAC2", "HDAC6", "HE2", "HE4", "HE5", "HEC1", "HEC1P", "HEK293", "HEK293T", "HEP3B", "HEPG2", "HERALPHA1", "HERALPHA2", "HEST2", "HFM1", "HIPK3", "HIRE1", "HMAD2", "HMG1", "HMGB1", "HN2", "HNF4", "HOP1", "HORMAD1", "HORMAD2", "HOX3", "HOXA1", "HOXA10", "HOXB1", "HOXB9", "HP1", "HPRT1", "HR6B", "HRK1", "HSF2", "HSP26", "HSP27", "HSP54", "HSP60", "HSP70", "HSP90", "HSPB1", "HT1A", "HT29", "HT2A", "HT2C", "HTR12", "HUH7", "HYAL1", "HYMA1", "I2", "I4", "IAP2", "IB4", "IC78", "ICI164", "IDH1", "IDH2", "IFI16", "IGF1", "IGF2", "IGF2R", "IGF3", "IL1B", "IL1RN", "ILE8", "IMA2", "IME4", "IMP1", "IMP2", "INSL3", "INSL6", "INSP3", "INSP3R", "INSP3RS", "INT1", "IP3", "IP3R", "IPLA2", "IR41A", "IR76B", "IRE1", "IRF3", "IRK1", "IRK2", "IRK3", "IRT1", "IRX3", "ISGF3", "ISL1", "J2", "JP8", "K10", "K12", "K123", "K16", "K195M", "K3", "K4", "K5", "K562", "K8", "K88", "K9", "KCC1", "KCC2", "KCNK5B", "KCNQ1", "KCNQ1OT", "KCNQ2", "KCNQ3", "KCNQ5", "KDM1A", "KDM2B", "KDM4B", "KDM5B", "KEAP1", "KEX2", "KIAA1429", "KIF11", "KIF20A", "KIN28", "KIP1", "KISS1", "KISS1R", "KISS2", "KLF4", "KLF6", "KRE33", "KVDMR1", "KZM1", "L09CD", "L1", "L4", "L5178Y", "L987A", "LAMBDA1", "LAMBDA2", "LAMBDA3", "LC3", "LD50", "LEFTY1", "LET7B", "LEU2", "LEU92", "LG12", "LG19", "LIM1", "LIS1", "LIV8", "LKB1", "LMP7", "LONDON1995", "LRP6", "LRRK2", "LSD1", "LSM1", "LSM10", "LSM11", "LY6", "LYS145", "LYS38", "M1", "M16", "M17", "M2", "M2A", "M3", "M3G", "M4", "M40I", "M5", "M5C", "M6", "M6A", "M7", "M8", "MAD2", "MAD2B", "MAGEA3", "MALAT1", "MAMMALS1", "MAP4", "MASH1", "MAT1", "MATRIX4", "MBD1", "MBD2", "MBD3", "MBD4", "MCD1P", "MCF7", "MCM7", "MCM8", "MCM9", "MCT1", "MCT10", "MCT2", "MCT3", "MCT4", "MCT8", "MDC15", "MDC9", "MDEG1", "MDEG2", "MDM2", "MDR1", "MDR1A", "MDR3", "MECP1", "MECP2", "MEF2", "MEI1", "MEK1", "MEK2", "MEK5", "MEK9", "MEL18", "MET15", "METTL14", "METTL3", "MG132", "MG1655", "MG2", "MGCL2", "MGLUR1", "MGLUR2", "MGLUR4A", "MGLUR5", "MGLUR7A", "MGLUR7B", "MGLUR8", "MGPX4", "MGSCV3", "MHCN1", "MHCN2", "MHCN4", "MICE1", "MIP1", "MIR160", "MIR164", "MIR17", "MIR19B", "MIR319", "MIR869", "ML171", "MLH1", "MLH3", "MLK3", "MMP10", "MMP3", "MN2", "MO15", "MODEL1", "MOP1", "MOP2", "MOP3", "MOP5", "MOUSE1", "MOV10", "MP1", "MPC1", "MPC11", "MPC2", "MPERIOD1", "MPP1", "MRE11", "MRF4", "MRP1", "MRP2", "MRP3", "MSE55", "MSH4", "MSH5", "MSI1", "MSIN3A", "MSX1", "MTA1S", "MTF2", "MTN3", "MUC1", "MUC16", "MUC17", "MUC2", "MUC3", "MURA3", "MYF5", "MYOD1", "MYT1", "MYT1HU", "MYT1XE", "N2A", "N40", "N6", "NAHCO3", "NANOS1", "NANOS2", "NANOS3", "NAT10", "ND5", "NDC80", "NDC80P", "NEDD1", "NEDD10", "NEDD2", "NEDD8", "NEK2", "NERVES9", "NEUROD1", "NEUROD2", "NEURONS1", "NF2", "NGPX4", "NH2", "NH4", "NH4CL", "NIC96", "NIP3", "NKX2", "NLRP1", "NLRP2", "NLRP4", "NLRP7", "NLRP9", "NMDAR1", "NMDAR2B", "NOTCH1", "NOX1", "NOX2", "NOX2DS", "NOX3", "NOX4", "NOX5", "NOXA1", "NOXO1", "NPH3", "NPM1", "NPR2", "NR1", "NR2", "NR2A", "NR2B", "NR2C", "NR2D", "NRAMP1", "NRAMP2", "NRF2", "NS2", "NTC12", "NTF2", "NTH15", "NUCLEI1", "NUF2", "NUF2P", "NUP107", "NUP153", "NUP50", "NUP93", "NUP96", "NUP98", "NVSR61", "O2", "O6", "OATP2", "OCT4", "ODF2", "OPA1", "ORB2", "ORC2", "ORF50", "ORF65", "OSD1", "OSR1", "OSTIR1", "OTUB1", "OTX2", "OVERVIEW2", "OX174", "OX513A", "P1", "P100", "P101L", "P102L", "P107", "P13", "P130", "P14", "P150", "P15O", "P16", "P160", "P170", "P185", "P19", "P2", "P204", "P21", "P22", "P23", "P27", "P2C", "P2X", "P3", "P300", "P301L", "P34", "P38", "P4", "P40", "P44", "P45", "P450", "P47", "P47V", "P48", "P5", "P50", "P53", "P54", "P55", "P57", "P60", "P60V", "P62", "P63", "P65", "P67", "P7", "P73", "P75", "P80", "P90", "P97", "PA28", "PA3552", "PA3559", "PAD4", "PAF1C", "PARK2", "PARK7", "PAT1", "PATTERN1", "PAX2", "PAX3", "PAX8", "PB1", "PB2", "PBS1", "PC1", "PC12", "PC2", "PC3", "PC6", "PCGF3", "PCH2", "PCM1", "PCO2", "PCV2", "PDCD1", "PDE3A", "PDE5", "PDK1", "PDK2", "PDK3", "PDK4", "PDS5", "PDSS2", "PEBP1", "PEBP2", "PEG1", "PEG10", "PEG2", "PEG3", "PELP1", "PER1", "PGAM5", "PGE1", "PGE2", "PGF2", "PGJ2", "PGRMC1", "PHE1", "PHE342", "PHE360", "PHERES1", "PI15", "PI16", "PI3", "PI31", "PI3K", "PIN1", "PINK1", "PIP1", "PIP1S", "PIP2", "PIWIL3", "PKD1", "PKD2", "PKHD1", "PKK223", "PKM1", "PKM2", "PKP3", "PLA2", "PLANTS1", "PLCZ1", "PLK1", "PLK4", "PLX1", "PMCA4", "POLR2A", "POU2", "POU91", "PP1", "PP2A", "PP60C", "PRAD1", "PRC2", "PRESSURE3", "PRM1", "PROK1", "PRRT2", "PRTN3", "PS1", "PS2", "PSP60", "PTPIP51", "PTRE1", "PTX3", "PXO99", "PYST1", "Q1", "Q10", "Q23", "R2", "R26R", "R277", "R3", "R3H", "R406W", "R98Q", "RAB11", "RAB35", "RAB5", "RAB7", "RAC1", "RAC2", "RACK1", "RAD1", "RAD2", "RAD21P", "RAD23", "RAD3", "RAD50", "RAD51", "RAD52", "RAD54", "RAD57", "RAD5O", "RAF1", "RANBP5", "RANBP7", "RAP1", "RAP1B", "RAP1P", "RAP55", "RAT1", "RATE6", "RB1", "RBF1", "RBF2", "RBM15", "RBM3", "RBM44", "REC8", "RED2", "REGIONS3", "RET1", "RGS10", "RGS11", "RGS3", "RGS4", "RGS5", "RGS6", "RGS7", "RGS8", "RGS9", "RLF2", "RNF212", "RNF212B", "RNH1", "RNS43", "RO60", "ROSA26", "ROUTES9", "RP1", "RPD3", "RPM1", "RRM1", "RRM2", "RRM3", "RSK1", "RSK2", "RSK3", "RSK4", "RUB1", "RUNX1", "RUNX2", "RYR1", "S0920", "S1", "S100", "S12", "S15", "S2", "S3", "S311A", "S349A", "S349D", "S6", "S652", "S7", "SA1", "SA2", "SALF17R", "SALL4", "SAMHD1", "SAP90", "SAR1", "SAS2", "SCC1", "SCC1P", "SCC3P", "SCCTR1", "SCCTR3", "SDP1", "SDP2", "SDP3", "SEC1", "SEC2", "SEED2", "SEEKER2", "SER325", "SER373", "SERPINA1", "SET7", "SETDB1", "SETH1", "SETH2", "SF1", "SF9", "SFRP2", "SFRP4", "SH2", "SH3", "SHANK3", "SINEB2", "SIP1", "SIR1", "SIR2", "SIR3", "SIR3P", "SIR4", "SIR4P", "SKF525A", "SKP1", "SLC2", "SLC30", "SLC36", "SLC39", "SLD5", "SLX4", "SMAD2", "SMAD4", "SMAD7", "SMARCAL1", "SMARCD1", "SMC1", "SMC1P", "SMC3", "SMC3P", "SMN1", "SMP2", "SNF2", "SOCIETY1986", "SOD1", "SOD2", "SOHLH1", "SOHLH2", "SOX1", "SOX11", "SOX2", "SOX8", "SOX9", "SOX9A", "SP1", "SP17", "SP3", "SP6", "SPACA1", "SPAG5", "SPATA16", "SPC97", "SPCAS9", "SPERM1", "SPIRE1", "SPIRE2", "SPLA2", "SPO11", "SPS1", "SRSF1", "SRSF3", "SRSF7", "STAG3", "STARD1", "STARD10", "STARD11", "STARD15", "STARD2", "STARD3", "STARD5", "STAT1", "STAT3", "STAT5", "STE20", "STK11", "STPK13", "SUC1", "SUL1", "SUL2", "SUV39H", "SV40", "SW3", "SW620", "SWI2", "SY5Y", "SYCE1", "SYCE2", "SYCP1", "SYCP2", "SYCP3", "SYNDROME1", "SYSTEM7", "T1", "T1689TS", "T2", "T2D", "T3", "T315A", "T4", "T40", "T46", "T47D", "T5", "T7", "T7EI", "T8", "TAB1", "TACC3", "TAK1", "TAN1", "TAP73", "TBX2", "TC10", "TCA3", "TDGF1", "TDMRT1", "TDRD12", "TEAD4", "TET1", "TET2", "TET3", "TEX14", "TG196", "TH0S", "TH1", "TH17", "TH1S", "TH2", "TH2B", "TH2S", "THR161", "THR167", "THR185", "THR233", "THUMPD1", "THY1", "TID3", "TIM3", "TIP1", "TIP2", "TIP60", "TIR1", "TLE6", "TLR2", "TLR4", "TLR5", "TM1", "TM10", "TM6", "TM8", "TMPRSS2", "TNFAIP6", "TOL2", "TOP1", "TOP1MT", "TOP2A", "TOP2B", "TOP3A", "TOP3B", "TOXICITY1", "TP2", "TP53", "TPC1", "TPC2", "TPX1", "TPX2", "TRAF1", "TREX1", "TRIP13", "TRP143", "TRP53", "TRP73", "TRPC3", "TRPC6", "TRPV1", "TRPV4", "TSG101", "TSG6", "TSP2", "TUBB4", "TUBB8", "TUT4", "TY1", "TYR694", "U1", "U13", "U2", "U4", "U5", "U6", "U7", "UBC1", "UBC12P", "UBC4", "UBC9P", "UBCH10", "UBE3A", "UBP8", "UCHL1", "UCP1", "UNIREF50", "UP1", "UPF1", "URA3", "USP2", "UWE25", "V174", "V1R", "V410L", "VAC1", "VANGL2", "VAPBP56S", "VAS2870", "VBCL2", "VITRO1", "VMD2", "VOLUME5", "VPS11", "VPS18", "VPS18P", "VPS4", "W8", "WEE1", "WEE2", "WH2", "WINDOWS95", "WNK1", "WNK4", "WNT1", "WNT8", "WT1", "X12", "X16", "X2", "XA13", "XBUB1", "XBUB3", "XCHK1", "XDNMT1", "XEEK1", "XERP1", "XKLP2", "XKR8", "XL2", "XLEG5", "XLPOU91", "XMAD1", "XMAD2", "XMAP215", "XNDC80", "XNR1", "XNR2", "XNR4", "XNR5", "XNR6", "XNUF2", "XP150", "XP22", "XP54", "XRAD21", "XRCC4", "XRS2", "XSA1", "XSA2", "XSMC1", "XSMC3", "XZW10", "Y705F", "YAP1", "YAP65", "YCF1", "YCF2", "YP3", "YTHDC1", "YTHDC2", "YTHDF1", "YTHDF2", "ZBP1", "ZF5", "ZIC5", "ZIF268", "ZK632", "ZMES1", "ZMES4", "ZN2", "ZNSO4", "ZP1", "ZP2", "ZP3", "ZPBP1", "ZPBP2"]
    graph = build_weighted_graph_from_string_files(
        genes=genes,
        info_path=SETTINGS.string_info_path,
        links_path=SETTINGS.string_links_path,
        required_score=SETTINGS.string_required_score,
        mode=SETTINGS.string_local_mode,
    )
    return {"graph": graph, "genes": genes}


def _graph_summary(graph: nx.Graph) -> dict[str, object]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    top = [{"gene": g, "degree": int(d)} for g, d in degrees[:10]]
    return {"nodes": n, "edges": m, "top_degree": top}


def node_rwr(state: AgentState) -> AgentState:
    graph = state.get("graph")
    genes = state.get("genes")
    nx.write_graphml(graph, "my_graph3.graphml")
    rwr = top_rwr_genes(graph, genes, top_k=100, restart_prob=0.1)
    return {"rwr_genes": rwr}


def node_enrichr(state: AgentState) -> AgentState:
    genes = state.get("genes")
    rwr = state.get("rwr_genes")
    expanded = genes + [g for g, _ in rwr]
    results = enrichr_pathways(
        expanded,
        top_n=10,
        background_genes=list((state.get("graph") or nx.Graph()).nodes()),
    )
    return {"enrichr": results}


def node_synthesize(state: AgentState) -> AgentState:
    query = state.get("query")
    genes = state.get("genes")
    graph = state.get("graph")
    rwr = state.get("rwr_genes")
    enrichr = state.get("enrichr")

    summary = _graph_summary(graph)
    answer = synthesize_technical_response(
        user_query=query,
        seed_genes=genes,
        rwr_genes=rwr,
        graph=graph,
        enrichr=enrichr,
    )

    meta = {
        "network": summary,
        "rwr_genes": rwr,
        "enrichr": enrichr,
    }
    return {"answer": answer, "meta": meta}


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("classify", node_classify)
    graph.add_node("general_answer", node_general_answer)

    graph.add_node("fetch_string", node_fetch_string)
    graph.add_node("rwr", node_rwr)
    graph.add_node("enrichr", node_enrichr)
    graph.add_node("synthesize", node_synthesize)

    graph.add_edge(START, "classify")
    # graph.add_edge(START, "fetch_string")  # skip classification for now
    graph.add_conditional_edges(
        "classify",
        _route,
        {
            "general": "general_answer",
            "technical": "fetch_string",
        },
    )

    graph.add_edge("general_answer", END)

    graph.add_edge("fetch_string", "rwr")
    graph.add_edge("rwr", "enrichr")
    graph.add_edge("enrichr", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()