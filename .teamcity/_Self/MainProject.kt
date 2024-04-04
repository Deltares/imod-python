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
import jetbrains.buildServer.configs.kotlin.projectFeatures.projectReportTab
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object MainProject : Project({
    buildType(Lint)
    buildType(MyPy)
    buildType(UnitTests)
    buildType(Examples)
    buildType(PipPython311)
    buildType(PipPython312)
    buildType(Tests)

    template(GitHubIntegrationTemplate)
    template(LintTemplate)
    template(MyPyTemplate)
    template(UnitTestsTemplate)
    template(ExamplesTemplate)
    template(PipPython310Template)
    template(PipPython311Template)
    template(PipPython312Template)

    features {
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

object PipPython310 : BuildType({
    name = "PipPython310"

    templates(PipPython310Template, GitHubIntegrationTemplate)
})

object PipPython311 : BuildType({
    name = "PipPython311"

    templates(PipPython311Template, GitHubIntegrationTemplate)
})

object PipPython312 : BuildType({
    name = "PipPython312"

    templates(PipPython312Template, GitHubIntegrationTemplate)
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
        snapshot(PipPython310) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(PipPython311) {
            onDependencyFailure = FailureAction.FAIL_TO_START
        }
        snapshot(PipPython312) {
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

