import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.CustomChart.Serie
import jetbrains.buildServer.configs.kotlin.CustomChart.SeriesKey
import jetbrains.buildServer.configs.kotlin.buildFeatures.*
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.projectFeatures.ProjectReportTab
import jetbrains.buildServer.configs.kotlin.projectFeatures.projectReportTab
import jetbrains.buildServer.configs.kotlin.triggers.finishBuildTrigger
import jetbrains.buildServer.configs.kotlin.triggers.schedule
import jetbrains.buildServer.configs.kotlin.triggers.vcs

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2023.11"

project {

    buildType(Lint)
    buildType(MyPy)
    buildType(UnitTests)
    buildType(Examples)
    buildType(Tests)

    template(GitHubIntegrationTemplate)
    template(LintTemplate)
    template(MyPyTemplate)
    template(UnitTestsTemplate)
    template(ExamplesTemplate)

    features {
        buildTypeCustomChart {
            id = "PROJECT_EXT_41"
            title = "Build Duration (all stages)"
            seriesTitle = "Serie"
            format = CustomChart.Format.DURATION
            series = listOf(
                Serie(title = "Build Duration (all stages)", key = SeriesKey.BUILD_DURATION)
            )
        }
        projectReportTab {
            id = "PROJECT_EXT_88"
            title = "Code Coverage"
            startPage = "coverage.zip!index.html"
            buildType = "iMOD6_IMODPython_Windows_Tests"
            sourceBuildRule = ProjectReportTab.SourceBuildRule.LAST_FINISHED
            sourceBuildBranchFilter = "+:<default>"
        }
    }

    subProject(Nightly)
    subProject(Pixi)
}

object GitHubIntegrationTemplate : Template({
    name = "GitHubIntegrationTemplate"

    features {
        commitStatusPublisher {
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "credentialsJSON:558df52e-822f-4d9d-825a-854846a9a2ff"
                }
            }
        }
        pullRequests {
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            provider = github {
                authType = token {
                    token = "credentialsJSON:558df52e-822f-4d9d-825a-854846a9a2ff"
                }
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
    }
})

object LintTemplate : Template({
    name = "LintTemplate"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
                name = "Static code analysis"
                id = "Static_code_analysis"
                workingDir = "imod-python"
                scriptContent = """
                    pixi run --frozen lint
                """.trimIndent()
                formatStderrAsError = true
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object MyPyTemplate : Template({
    name = "MyPyTemplate"

    detectHangingBuilds = false

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
                name = "MyPy analysis"
                id = "MyPy_analysis"
                workingDir = "imod-python"
                scriptContent = """
                    pixi run --frozen mypy_report
                    pixi run --frozen mypy
                """.trimIndent()
                formatStderrAsError = true
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/*.xml"
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object UnitTestsTemplate : Template({
    name = "UnitTestsTemplate"

    allowExternalStatus = true
    artifactRules = """
        imod-python\imod\tests\temp => test_output.zip
        imod-python\imod\tests\coverage => coverage.zip
    """.trimIndent()

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Run unittests"
            id = "Run_unittests"
            workingDir = "imod-python"
            scriptContent = """
                set Path=%system.teamcity.build.checkoutDir%\modflow6;%env.Path% 
                pixi run --frozen unittests
            """.trimIndent()
            formatStderrAsError = true
        }
        powerShell {
            name = "Extract coverage statistics"
            id = "Extract_coverage_statistics"
            workingDir = "imod-python/imod/tests"
            scriptMode = script {
                content = """
                    ${'$'}REPORT = echo "coverage report" | pixi shell
                    
                    ${'$'}TOTALS = ${'$'}REPORT | Select-String -Pattern 'TOTAL' -CaseSensitive -SimpleMatch
                    ${'$'}STATISTICS = ${'$'}TOTALS -split "\s+"
                    ${'$'}TOTALLINES = ${'$'}STATISTICS[1]
                    ${'$'}MISSEDLINES = ${'$'}STATISTICS[2]
                    ${'$'}COVEREDLINES = ${'$'}TOTALLINES - ${'$'}MISSEDLINES
                    
                    Write-Host "##teamcity[buildStatisticValue key='CodeCoverageAbsLCovered' value='${'$'}COVEREDLINES']"
                    Write-Host "##teamcity[buildStatisticValue key='CodeCoverageAbsLTotal' value='${'$'}TOTALLINES']"
                """.trimIndent()
            }
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/imod/tests/*report.xml"
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object ExamplesTemplate : Template({
    name = "ExamplesTemplate"

    artifactRules = """imod-python\imod\tests\temp => test_output.zip"""

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")
        root(AbsoluteId("iMOD6_IMODPython_MetaSwapLookupTable"), ". => lookup_table")

        cleanCheckout = true
    }

    steps {
        script {
            name = "Run examples"
            id = "Run_examples"
            workingDir = "imod-python"
            scriptContent = """
                set Path=%system.teamcity.build.checkoutDir%\modflow6;%env.Path% 
                pixi run --frozen examples
            """.trimIndent()
            formatStderrAsError = true
        }
    }

    features {
        xmlReport {
            reportType = XmlReport.XmlReportType.JUNIT
            rules = "imod-python/imod/tests/*report.xml"
        }
    }

    requirements {
        equals("env.OS", "Windows_NT")
    }
})

object Lint : BuildType({
    name = "Lint"

    templates(LintTemplate, GitHubIntegrationTemplate)
})

object MyPy : BuildType({
    name = "MyPy"

    templates(MyPyTemplate, GitHubIntegrationTemplate)

    failureConditions {
        nonZeroExitCode = false
        testFailure = false
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.TEST_FAILED_COUNT
            threshold = 26
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.MORE
            compareTo = value()
        }
    }
})

object UnitTests : BuildType({
    name = "UnitTests"

    templates(UnitTestsTemplate, GitHubIntegrationTemplate)

    dependencies {
        dependency(AbsoluteId("MetaSWAP_Modflow_Modflow6Release642")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
        snapshot(Lint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(MyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
    }
})

object Examples : BuildType({
    name = "Examples"

    templates(ExamplesTemplate, GitHubIntegrationTemplate)

    dependencies {
        dependency(AbsoluteId("MetaSWAP_Modflow_Modflow6Release642")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
        snapshot(Lint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(MyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
    }
})

object Tests : BuildType({
    name = "Tests"

    allowExternalStatus = true
    type = Type.COMPOSITE

    vcs {
        root(DslContext.settingsRoot)

        branchFilter = """
            +:*
            -:release_imod56
        """.trimIndent()
        showDependenciesChanges = true
    }

    triggers {
        vcs {
        }
    }

    features {
        pullRequests {
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            provider = github {
                authType = token {
                    token = "credentialsJSON:558df52e-822f-4d9d-825a-854846a9a2ff"
                }
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
    }

    dependencies {
        snapshot(Examples) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(UnitTests) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:coverage.zip => ."
            }
        }
    }
})

object Nightly : Project({
    name = "Nightly"

    buildType(NightlyLint)
    buildType(NightlyMyPy)
    buildType(NightlyUnitTests)
    buildType(NightlyExamples)
    buildType(NightlyTests)
})

object NightlyLint : BuildType({
    name = "Lint"

    templates(LintTemplate)
})

object NightlyMyPy : BuildType({
    name = "MyPy"

    templates(MyPyTemplate)
})

object NightlyUnitTests : BuildType({
    name = "UnitTests"

    templates(UnitTestsTemplate)

    dependencies {
        snapshot(NightlyLint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(NightlyMyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(AbsoluteId("iMOD6_Modflow6buildWin64")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
    }
})

object NightlyExamples : BuildType({
    name = "Examples"

    templates(ExamplesTemplate)

    dependencies {
        snapshot(NightlyLint) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(NightlyMyPy) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(AbsoluteId("iMOD6_Modflow6buildWin64")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
    }
})

object NightlyTests : BuildType({
    name = "Tests"

    allowExternalStatus = true
    type = Type.COMPOSITE

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"))

        branchFilter = """
            +:*
            -:release_imod56
        """.trimIndent()
        showDependenciesChanges = true
    }

    triggers {
        finishBuildTrigger {
            buildType = "iMOD6_Modflow6buildWin64"
            successfulOnly = true
        }
    }

    features {
        notifications {
            notifierSettings = emailNotifier {
                email = """
                    joeri.vanengelen@deltares.nl
                    luitjan.slooten@deltares.nl
                    sunny.titus@deltares.nl
                """.trimIndent()
            }
            buildFailedToStart = true
            buildFailed = true
        }
    }

    dependencies {
        snapshot(NightlyExamples) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        dependency(NightlyUnitTests) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:coverage.zip => ."
            }
        }
    }
})

object Pixi : Project({
    name = "Pixi"

    buildType(UpdateDependencies)
})

object UpdateDependencies : BuildType({
    name = "Update Dependencies"

    params {
        param("GH_USER", "deltares-service-account")
        text("env.GH_TOKEN", "%github_deltares-service-account_access_token%")
    }

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
    }

    steps {
        powerShell {
            name = "Update dependencies"
            id = "Update_dependencies"
            workingDir = "imod-python"
            scriptMode = script {
                content = """
                    echo "Create update branch"
                    git remote set-url origin https://%GH_USER%:%env.GH_TOKEN%@github.com/Deltares/imod-python.git
                    git checkout -b pixi_update_%build.counter%
                    
                    echo "Update dependencies" 
                    del pixi.lock
                    pixi install
                    
                    echo "Add any changes" 
                    git add pixi.lock
                    
                    if (git status -suno) 
                    {
                      git commit -m "Update pixi.lock"
                      git push -u origin pixi_update_%build.counter%
                      pixi run gh pr create --title "[TEAMCITY] Update project dependencies" --body "Teamcity automatically updated the dependencies defined the pixi.toml file. Please verify that all tests succeed before merging" --reviewer JoerivanEngelen,luitjansl
                      echo "Changes pushed and PR created"
                    }
                    else
                    {
                      echo "No changes found"
                    }
                """.trimIndent()
            }
            noProfile = false
        }
    }

    triggers {
        schedule {
            schedulingPolicy = weekly {
                hour = 15
            }
            branchFilter = "+:<default>"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    failureConditions {
        errorMessage = true
    }
})
