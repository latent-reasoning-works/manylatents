rm(list = ls())

library('parallel')

#chromosomes = c(22)
chromosomes = c(seq(1,22))

options(width = 180)

#storageDirectory = '/home/jupyter/workspaces/geneticancestry/Data/'
# mkdir /home/jupyter/workspaces/phaterepresentationsforvisualizationofgeneticdata/Data/1KGPHGDPAOU
storageDirectory = '/home/jupyter/workspaces/phaterepresentationsforvisualizationofgeneticdata/V2/Data/'

combineGenotypes = function(chromosome = NULL, runningSet = NULL, addInSet = NULL, fillAddIn = FALSE) {

    new = paste0(runningSet, addInSet)
    outputDirectory = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/')
    
    ## First try to see if we can find a common set of SNPs between the two datasets.  Rename sites from the add-in set if needed.
    runningPrefix = paste0(storageDirectory, runningSet, '/extractedChr', chromosome)
    runningBIMFile = paste0(runningPrefix, '.bim')
    runningBIM = read.table(runningBIMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
    addInPrefix = paste0(storageDirectory, addInSet, '/extractedChr', chromosome)
    addInBIMFile = paste0(addInPrefix, '.bim')
    addInBIM = read.table(addInBIMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
    
    runningInAddInBySite = runningBIM[ , 4] %in%  addInBIM[ , 4]
    addInInRunningBySite = addInBIM[ , 4] %in%  runningBIM[ , 4]
    runningInAddInByID = runningBIM[ , 2] %in%  addInBIM[ , 2]
    addInInRunningByID = addInBIM[ , 2] %in%  runningBIM[ , 2]
    
    print(paste('Running in AddIn by site:', sum(runningInAddInBySite), '/', length(runningInAddInBySite)))
    print(paste('AddIn in Running by site:', sum(addInInRunningBySite), '/', length(addInInRunningBySite)))
    print(paste('Running in AddIn by ID:', sum(runningInAddInByID), '/', length(runningInAddInByID)))
    print(paste('AddIn in Running by ID:', sum(addInInRunningByID), '/', length(addInInRunningByID)))


    ## Give the addIn set ID's from the running set
    siteLookup = runningBIM[runningInAddInBySite, 2]
    names(siteLookup) = as.character(runningBIM[runningInAddInBySite, 4])
    addInBIM[addInInRunningBySite, 2] = siteLookup[as.character(addInBIM[addInInRunningBySite, 4])]

    runningInAddInBySite = runningBIM[ , 4] %in%  addInBIM[ , 4]
    addInInRunningBySite = addInBIM[ , 4] %in%  runningBIM[ , 4]
    runningInAddInByID = runningBIM[ , 2] %in%  addInBIM[ , 2]
    addInInRunningByID = addInBIM[ , 2] %in%  runningBIM[ , 2]
    
    print(paste('Running in AddIn by site:', sum(runningInAddInBySite), '/', length(runningInAddInBySite)))
    print(paste('AddIn in Running by site:', sum(addInInRunningBySite), '/', length(addInInRunningBySite)))
    print(paste('Running in AddIn by ID:', sum(runningInAddInByID), '/', length(runningInAddInByID)))
    print(paste('AddIn in Running by ID:', sum(addInInRunningByID), '/', length(addInInRunningByID)))

    ## De-duplicate both sets
    uniqueRunningBIM = unique(runningBIM, MAR = 1)
    uniqueAddInBIM = unique(addInBIM, MAR = 1)
    print(dim(addInBIM))
    print(dim(uniqueAddInBIM))

    runningBIM[-match(unique(runningBIM[,2]), runningBIM[,2]), 2] = 'REPEAT'
    addInBIM[-match(unique(addInBIM[,2]), addInBIM[,2]), 2] = 'REPEAT'

    runningInAddInBySite = uniqueRunningBIM[ , 4] %in%  uniqueAddInBIM[ , 4]
    addInInRunningBySite = uniqueAddInBIM[ , 4] %in%  uniqueRunningBIM[ , 4]
    runningInAddInByID = uniqueRunningBIM[ , 2] %in%  uniqueAddInBIM[ , 2]
    addInInRunningByID = uniqueAddInBIM[ , 2] %in%  uniqueRunningBIM[ , 2]
    
    print(paste('Running in AddIn by site:', sum(runningInAddInBySite), '/', length(runningInAddInBySite)))
    print(paste('AddIn in Running by site:', sum(addInInRunningBySite), '/', length(addInInRunningBySite)))
    print(paste('Running in AddIn by ID:', sum(runningInAddInByID), '/', length(runningInAddInByID)))
    print(paste('AddIn in Running by ID:', sum(addInInRunningByID), '/', length(addInInRunningByID)))

    ## Grab just the common sites
    print(dim(uniqueAddInBIM))
    uniqueAddInBIM = uniqueAddInBIM[addInInRunningByID, ]
    print(dim(uniqueAddInBIM))
    
    ## Extract the de-duplicated, common sites from each.
    addInBIMFile = paste0(outputDirectory, 'addInChr', chromosome, '.bim')
    write.table(x = addInBIM, file = addInBIMFile, row.names = FALSE, col.names = FALSE, quote = FALSE)
    uniqueAddInPrefix = paste0(outputDirectory, 'uniqueAddInChr', chromosome) 
    uniqueAddInBIMFile = paste0(uniqueAddInPrefix, '.bim')
    write.table(x = uniqueAddInBIM, file = uniqueAddInBIMFile, row.names = FALSE, col.names = FALSE, quote = FALSE)
    print(uniqueAddInBIMFile)
    command = paste('plink',
        '--keep-allele-order',
        '--allow-no-sex',
        '--bed', paste0(addInPrefix, '.bed'),
        '--bim', addInBIMFile,
        '--fam', paste0(addInPrefix, '.fam'),
        '--extract', uniqueAddInBIMFile,
        '--make-bed',
        '--chr', chromosome,
        '--out', uniqueAddInPrefix)
    if (!fillAddIn) {
        command = paste(command, '--geno 0.05')
    }
    print(command)
    system(command)

    ## Make a new file from the running dataset.  If we aren't filling in the addin set, then keep just the common SNPs
    runningBIMFile = paste0(outputDirectory, 'runningChr', chromosome, '.bim')
    write.table(x = runningBIM, file = runningBIMFile, row.names = FALSE, col.names = FALSE, quote = FALSE)
    uniqueRunningPrefix = paste0(outputDirectory, 'uniqueRunningInChr', chromosome) 
    command = paste('plink',
        '--keep-allele-order',
        '--allow-no-sex',
        '--bed', paste0(runningPrefix, '.bed'),
        '--bim', runningBIMFile,
        '--fam', paste0(runningPrefix, '.fam'),
        '--make-bed',
        '--chr', chromosome,
        '--out', uniqueRunningPrefix)
    if (fillAddIn) {
        command = paste(command, '--extract', runningBIMFile)
    } else {
        command = paste(command, '--extract', uniqueAddInBIMFile)
    }
    print(command)
    system(command)
    
    runningPrefix = uniqueRunningPrefix
    addInPrefix = uniqueAddInPrefix
    
    ## Try to merge two datasets.  This might fail because alignments are not correct all the time.
    outputPrefix = paste0(outputDirectory, '/extractedChr', chromosome)
    command = paste('plink',
        '--make-bed',
        '--bfile', runningPrefix,
        '--bmerge', addInPrefix,
        '--out', outputPrefix,
        '--keep-allele-order',
        '--allow-no-sex')
    print(command)
    system(command)

    ## Assume that merging went right
    rightPrefix = addInPrefix
    rightBIMFile = paste0(rightPrefix, '.bim')

    addInBIM = paste0(addInPrefix, '.bim')
    addInBIM = read.table(file = addInBIMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
    
    atgcSites = (addInBIM[ , 5] == 'A' & addInBIM[ , 6] == 'T') |
        (addInBIM[ , 5] == 'T' & addInBIM[ , 6] == 'A') |
            (addInBIM[ , 5] == 'G' & addInBIM[ , 6] == 'C') |
                (addInBIM[ , 5] == 'C' & addInBIM[ , 6] == 'G')
    print(paste('AT-GC sites:', sum(atgcSites)))
    
    atgcBIM = addInBIM[atgcSites, ]
    atgcBIMFile = paste0(outputDirectory, 'atgcChr', chromosome, '.bim')
    write.table(file = atgcBIMFile, x = atgcBIM, row.names = FALSE, col.names = FALSE, quote = FALSE)
    
    ##See if there is a missnp file.  There should be.  If there is, try to deal with those SNPs, i.e., flip them, then try again
    misSNPFile = paste(outputPrefix, '-merge.missnp', sep = '')
    if (!is.na(file.info(misSNPFile)[1,'size'])) {

        misSNPs = read.table(file = misSNPFile, stringsAsFactors = FALSE, row.names = NULL, header = FALSE)[,1]
        system(paste('rm', misSNPFile))
        
        runningBIMFile = paste0(runningPrefix, '.bim')
        runningBIM = read.table(file = runningBIMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
                
        misSNPSites = addInBIM[ , 2] %in% misSNPs
        print(paste('Missnp sites:', sum(misSNPSites)))

        nonATGCMisSNPSites = misSNPSites & !atgcSites
        print(paste('Non AT-GC missnp sites:', sum(nonATGCMisSNPSites)))
        
        flipBIM = addInBIM[nonATGCMisSNPSites, ]
        flipBIMFile = paste0(outputDirectory, 'flipChr', chromosome, '.bim')
        write.table(file = flipBIMFile, x = flipBIM, row.names = FALSE, col.names = FALSE, quote = FALSE)
        
        rightPrefix = paste0(outputDirectory, 'rightChr', chromosome)
        rightBIMFile = paste0(rightPrefix, '.bim')
        command = paste('plink',
            '--make-bed',
            '--bfile', addInPrefix,
            #'--extract', runningBIMFile,
            '--flip', flipBIMFile,
            #'--exclude', flipBIMFile,
            '--out', rightPrefix,
            '--keep-allele-order',
            '--allow-no-sex')
        print(command)
        system(command)
    }        
        
    leftPrefix = paste0(outputDirectory, '/leftChr', chromosome)
    leftBIMFile = paste0(leftPrefix, '.bim')
    command = paste('plink',
        '--keep-allele-order',
        '--allow-no-sex',
        '--make-bed',
        '--bfile', runningPrefix,
        '--out', leftPrefix)
    if (!fillAddIn) {
        command = paste(command, '--extract', rightBIMFile)
    }
    print(command)
    system(command)
        
        
    command = paste('plink',
        '--make-bed',
        '--bfile', leftPrefix,
        '--bmerge', rightPrefix,
        '--out', outputPrefix,
        '--chr', chromosome,
        '--keep-allele-order',
        '--allow-no-sex')
    if (!fillAddIn) {
        command = paste(command, '--geno 0.05')
    }

    print(command)
    system(command)
        
    misSNPFile = paste(outputPrefix, '-merge.missnp', sep = '')
    ## And again, because any SNPs that didn't flip are genuinely bad between the files
    if (!is.na(file.info(misSNPFile)[1,'size'])) {        
        command = paste('plink',
            '--keep-allele-order',
            '--allow-no-sex',
            '--bfile', leftPrefix,
            '--exclude', misSNPFile,
            '--make-bed',
            '--out', leftPrefix)
        print(command)
        system(command)
    
    
        command = paste('plink',
            '--keep-allele-order',
            '--allow-no-sex',
            '--bfile', rightPrefix,
            '--exclude', misSNPFile,
            '--make-bed',
            '--out', rightPrefix)
        print(command)
        system(command)
    }
            
    command = paste('plink',
        '--make-bed',
        '--bfile', leftPrefix,
        '--bmerge', rightPrefix,
        '--out', outputPrefix,
        '--chr', chromosome,
        '--keep-allele-order',
        '--allow-no-sex')
    if (!fillAddIn) {
        command = paste(command, '--geno 0.05')
    }
    print(command)
    system(command)
    
    ## Do a flip scan of the AT-GC SNPs.  Toss potentially flipped SNPs
    flipPrefix = paste0(outputDirectory, '/flipping', chromosome)
    mergedFAMFile = paste0(outputPrefix, '.fam')
    mergedFAM = read.table(file = mergedFAMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
    flipFAMFile = paste0(flipPrefix, '.fam')
    leftPops = read.table(file = paste0(leftPrefix, '.fam'),
        row.names = NULL, header = FALSE, stringsAsFactors = FALSE)[,1]
    rightPops = read.table(file = paste0(rightPrefix, '.fam'),
        row.names = NULL, header = FALSE, stringsAsFactors = FALSE)[,1]
    flipFAM = mergedFAM
    flipFAM[,6] = 1
    flipFAM[flipFAM[ , 1] %in% rightPops, 6] = 2
    flipFAM[ , c(3,4,5)] = 0
    write.table(x = flipFAM, file = flipFAMFile, row.names = FALSE, col.names = FALSE, quote = FALSE)
    command = paste('plink',
        '--keep-allele-order',
        '--bed', paste0(outputPrefix, '.bed'),
        '--bim', paste0(outputPrefix, '.bim'),
        '--fam', flipFAMFile,
        '--flip-scan',
        '--allow-no-sex',
        '--out', flipPrefix)
    print(command)
    system(command)
    
    flipscanFile = paste0(flipPrefix, '.flipscan')
    flipscan = read.table(file = flipscanFile, header = TRUE, row.names = NULL, stringsAsFactors = FALSE, fill = TRUE)
    flipscan = flipscan[flipscan[,7] <= flipscan[,9] & flipscan[,9] > 1,]
    if (nrow(flipscan) > 0) {
        flipscan = cbind(flipscan[,1:2], 0, flipscan[,3])
        flipscan = flipscan[flipscan[ , 2] %in% atgcBIM[ , 2], ]
        exclude = paste0(flipPrefix, '.exclude')
        write.table(file = exclude, x = flipscan, row.names = FALSE, col.names= FALSE, quote = FALSE)
    }
    command = paste('plink',
        '--keep-allele-order',
        '--bfile', outputPrefix,
        '--make-bed',
        '--out', outputPrefix)
    if (fillAddIn) {
        command = paste(command, '--fill-missing-a2')
    }
    if (nrow(flipscan) > 0) {
        command = paste(command, '--exclude', exclude)
    }
    print(command)
    system(command)
}

mergeChromosomes = function(runningSet, addInSet, fstFilter = FALSE, fillAddIn = FALSE) { 

    ## Merge each chromosome
    mclapply(chromosomes, function(i) combineGenotypes(i, runningSet, addInSet, fillAddIn = fillAddIn), mc.cores = 22)

    ## And merge the chromosomes
    outputPrefix = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/extractedChr')    
    mergeListFile = paste0(outputPrefix, 'All.merge')
    mergeList = cbind(paste0(outputPrefix, chromosomes, '.bed'),
        paste0(outputPrefix, chromosomes, '.bim'),
        paste0(outputPrefix, chromosomes, '.fam'))
    write.table(x = mergeList, file = mergeListFile, quote = FALSE, row.names = FALSE, col.names = FALSE)
    mergedPrefix = paste0(outputPrefix, 'AllUnpruned')
    misSNPFile = paste0(mergedPrefix, '-merge.missnp')

    command = paste('plink',
        '--keep-allele-order',
        '--merge-list', mergeListFile,
        '--make-bed',
        '--geno', 0.01,
        '--maf 0.001',
        '--allow-no-sex',
        '--out', mergedPrefix)
    print(command)
    system(command)
             
    ##See if we have a missnp file
    if (!is.na(file.info(misSNPFile)[1,'size'])) {
        
        filteredPrefix = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/filteredExtractedChr')
        
        filterFunc = function(i) {
            iInPrefix = paste0(outputPrefix, i);
            iOutPrefix = paste0(filteredPrefix, i);
            command = paste('plink',
                '--keep-allele-order',
                '--bfile', iInPrefix,
                '--make-bed',
                '--exclude', misSNPFile,
                '--out', iOutPrefix,
                '--geno 0.01',
                '--allow-no-sex');
            print(command);
            system(command)
        }
        mclapply(chromosomes, filterFunc, mc.cores = 22)

        mergeList = cbind(paste0(filteredPrefix, chromosomes, '.bed'),
            paste0(filteredPrefix, chromosomes, '.bim'),
            paste0(filteredPrefix, chromosomes, '.fam'))
        write.table(x = mergeList, file = mergeListFile, quote = FALSE, row.names = FALSE, col.names = FALSE)
        command = paste('plink',
            '--keep-allele-order',
            '--merge-list', mergeListFile,
            '--make-bed',
            '--geno', 0.01,
            '--allow-no-sex',
            '--out', mergedPrefix)
        print(command)
        system(command)
    }

    ## Remove indels because biallelic SNPs can confidently determine genetic ancestry.
    bimFile = paste0(mergedPrefix, '.bim')
    indelBIMFile = paste0(mergedPrefix, 'indel')
    command = paste('cat', bimFile,
        '| awk \'length($5) != 1 || length($6) != 1\'',
        '>', indelBIMFile)
    print(command)
    system(command)

    noIndelPrefix = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/noIndelExtractedChrAll')
    
    command = paste('plink',
        '--keep-allele-order',
        '--bfile', mergedPrefix,
        '--exclude', indelBIMFile,
        '--make-bed', 
        '--out', noIndelPrefix)
    print(command)
    system(command)

    finalPrefix = noIndelPrefix
    
    ## Perform Fst pruning and remove variants that do not passs the filter.
    if (fstFilter) {

        ## First, find the addin populations
        addInFAMFile = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/uniqueAddInChr1.fam')
        
        addInFAM = read.table(file = addInFAMFile, row.names = NULL, header = FALSE, stringsAsFactors = FALSE)
        addInPops = unique(addInFAM[ , 1])
        addInRegex = paste0('/', paste(addInPops, collapse = '|'), '/')

        famFile = paste0(noIndelPrefix, '.fam')
        withinFile = paste0(noIndelPrefix, '.within')
        
        command = paste('cat', famFile,
            '| awk \'{g = ""; if($1 ~ /./){g = "running"};',
            'if($1 ~', addInRegex, '){g = "addin"}OFS = "\t";if(g != ""){print $1,$2,g}}\'',
            '>', withinFile)
        print(command)
        system(command)
        
        command = paste('plink --fst',
            '--keep-allele-order',
            '--bfile', noIndelPrefix,
            '--within', withinFile,
            '--out', noIndelPrefix)
        print(command)
        system(command)
        
        fstFile = paste0(noIndelPrefix, '.fst')
        fst = read.table(file = fstFile, header = TRUE, row.names = NULL, stringsAsFactors = FALSE)
        fst[is.nan(fst[,'FST']), 'FST'] = 0
        highFst = fst[fst[,'FST'] >= 0.15, ]
        highFst = highFst[complete.cases(highFst), , drop = FALSE]
        if (nrow(highFst) > 0) {
            highFst = cbind(highFst[,c(1,2)], 0, highFst[,3])
            highFstFile = paste0(noIndelPrefix, 'High.tsv')
            write.table(x = highFst, file = highFstFile, col.names = FALSE, row.names = FALSE, quote = FALSE)
            
            fstPrunedPrefix = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/fstPrunedChrAll')

            command = paste('plink',
                '--keep-allele-order',
                '--bfile', noIndelPrefix,
                '--exclude', highFstFile,
                '--make-bed',
                '--out', fstPrunedPrefix)
            print(command)
            system(command)
            
            finalPrefix = fstPrunedPrefix
        }
    }

    ## Move whatever was done to the output prefix
    command = paste('plink',
        '--keep-allele-order',
        '--allow-no-sex',
        '--bfile', finalPrefix,
        '--make-bed',
        '--out', mergedPrefix)
    print(command)
    system(command)
             
    ##############################################
    ###You can also consider HWE pruning here.###
    ##############################################
             
    ## And prune for reducing data size for genetic ancestry inference.
    prunedPrefix = paste0(storageDirectory, '1KGPHGDPAOU_V7', '/extractedChrAllPruned')
    
    command = paste('plink',
        '--keep-allele-order',
        '--maf 0.01',
        '--indep-pairwise 50 10 .1',
        '--make-bed',
        '--allow-no-sex',
        '--bfile', mergedPrefix,
        '--out', prunedPrefix)
    print(command)
    system(command)

    command = paste('plink',
        '--keep-allele-order',
        '--allow-no-sex',
        '--make-bed',
        '--extract', paste0(prunedPrefix, '.prune.in'),
        '--bfile', mergedPrefix,
        '--out', prunedPrefix)
    print(command)
    system(command)
    
    ## Also split it up in case we need it later
    for (chr in chromosomes) {
        command = paste('plink',
            '--keep-allele-order',
            '--allow-no-sex',
            '--bfile', mergedPrefix,
            '--make-bed',
            '--chr', chr,
            '--out', paste0(storageDirectory, '1KGPHGDPAOU_V7', '/extractedChr', chr))

        print(command)
        system(command)
    }    
}

mergeChromosomes(runningSet='1KGPHGDP', addInSet='AllofUs_V7', fstFilter=TRUE)