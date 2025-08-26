package _Self

import Deploy.DeployProject
import Nightly.NightlyProject
import Pixi.PixiProject
import Templates.*
import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.projectFeatures.ProjectReportTab
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry
import jetbrains.buildServer.configs.kotlin.projectFeatures.projectReportTab
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object MainProject : Project({
    params {
        param("DockerContainer", "containers.deltares.nl/hydrology_product_line_imod/windows-pixi")
        param("DockerVersion", "v0.39.2")
    }

    buildType(Lint)
    buildType(MyPy)
    buildType(UnitTests)
    buildType(Examples)
    buildType(PipPython)
    buildType(Tests)

    template(GitHubIntegrationTemplate)
    template(LintTemplate)
    template(MyPyTemplate)
    template(UnitTestsTemplate)
    template(ExamplesTemplate)
    template(PipPythonTemplate)

    features {
        dockerRegistry {
            id = "PROJECT_EXT_134"
            name = "Hydrology"
            url = "https://containers.deltares.nl/"
            userName = "robot${'$'}hydrology_product_line_imod+teamcity"
            password = "credentialsJSON:7cfeee0c-bc26-4c80-b488-a5d8e00233c3"
        }
        buildTypeCustomChart {
            id = "PROJECT_EXT_41"
            title = "Build Duration (all stages)"
            seriesTitle = "Serie"
            format = CustomChart.Format.DURATION
            series = listOf(
                CustomChart.Serie(title = "Build Duration (all stages)", key = CustomChart.SeriesKey.BUILD_DURATION)
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

    subProject(DeployProject)
    subProject(NightlyProject)
    subProject(PixiProject)
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
            threshold = 0
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.MORE
            compareTo = value()
        }
    }
})

object UnitTests : BuildType({
    name = "UnitTests"

    params {
        param("reverse.dep.Modflow_Modflow6Release.MODFLOW6_Version", "6.6.2")
        param("reverse.dep.Modflow_Modflow6Release.MODFLOW6_Platform", "win64")
    }
    templates(UnitTestsTemplate, GitHubIntegrationTemplate)

    dependencies {
        dependency(AbsoluteId("Modflow_Modflow6Release")) {
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

    params {
        param("reverse.dep.Modflow_Modflow6Release.MODFLOW6_Version", "6.6.2")
        param("reverse.dep.Modflow_Modflow6Release.MODFLOW6_Platform", "win64")
    }

    templates(ExamplesTemplate, GitHubIntegrationTemplate)

    dependencies {
        dependency(AbsoluteId("Modflow_Modflow6Release")) {
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

object PipPython : BuildType({
    name = "PipPython"

    templates(PipPythonTemplate, GitHubIntegrationTemplate)
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
        snapshot(PipPython) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
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

