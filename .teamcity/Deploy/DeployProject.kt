package Deploy

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object DeployProject : Project({
    name = "Deploy"

    buildType(BuildPackage)
    buildType(BuildPages)
    buildType(CreateGitHubRelease)
    buildType(DeployPackage)
    buildType(DeployPages)
    buildType(DeployAll)
})

object BuildPackage : BuildType({
    name = "Build Package"

    artifactRules = """imod-python\dist => dist.zip"""

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
    }

    steps {
        script {
            name = "Create package"
            id = "Create_package"
            workingDir = "imod-python"
            scriptContent = """
                SET TEMP=%system.teamcity.build.checkoutDir%\tmpdir
                SET TMPDIR=%system.teamcity.build.checkoutDir%\tmpdir
                SET TMP=%system.teamcity.build.checkoutDir%\tmpdir

                pixi run --environment default --frozen rm --recursive --force dist
                pixi run --environment default --frozen python -m build
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=4 --memory=16g"""
            dockerPull = false
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})

object BuildPages : BuildType({
    name = "Build Pages"

    artifactRules = """imod-python\docs\_build\html => documentation.zip"""

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_MetaSwapLookupTable"), ". => lookup_table")
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
    }

    steps {
        script {
            name = "Build documentation"
            workingDir = "imod-python"
            scriptContent = """
                SET PATH=%%PATH%%;%system.teamcity.build.checkoutDir%\modflow6
                
                pixi run --environment default --frozen docs
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=8 --memory=32g"""
            dockerPull = false
        }
    }

    triggers {
        vcs {
            enabled = false
        }
    }

    dependencies {
        dependency(AbsoluteId("MetaSWAP_Modflow_Modflow6Release642")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                buildRule = lastSuccessful()
                artifactRules = "+:MODFLOW6.zip!** => modflow6"
            }
        }
        artifacts(AbsoluteId("iMOD6_Modflow6buildWin64")) {
            artifactRules = """
                src/mf6.exe => modflow6/
            """.trimIndent()
            enabled = false
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})

object CreateGitHubRelease : BuildType({
    name = "Create GitHub release"

    enablePersonalBuilds = false
    type = Type.DEPLOYMENT
    maxRunningBuilds = 1

    params {
        param("env.GH_TOKEN", "%github_deltares-service-account_access_token%")
        param("env.PIXI_BETA_WARNING_OFF", "true")
    }

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
    }

    steps {
        powerShell {
            name = "Create GitHub release"
            id = "Create_GitHub_release"
            formatStderrAsError = true
            workingDir = "imod-python"
            scriptMode = script {
                content = """
                    ${'$'}tag = git describe --tags --abbrev=0 --exact-match
                    
                    echo "Creating GitHub release for: ${'$'}tag"
                    pixi run --environment default --frozen gh release create ${'$'}tag --verify-tag --notes "See https://deltares.github.io/imod-python/api/changelog.html"
                """.trimIndent()
            }
            param("plugin.docker.imagePlatform", "windows")
            param("plugin.docker.pull.enabled", "false")
            param("plugin.docker.imageId", "%DockerContainer%:%DockerVersion%")
            param("plugin.docker.run.parameters", "--cpus=4 --memory=16g")
        }
    }

    failureConditions {
        errorMessage = true
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})

object DeployPackage : BuildType({
    name = "Deploy Package"

    params {
        param("env.TWINE_USERNAME", "__token__")
        param("env.TWINE_NON_INTERACTIVE", "true")
        param("env.PIXI_BETA_WARNING_OFF", "true")
        password("env.TWINE_PASSWORD", "credentialsJSON:2cea585c-e4f8-4a45-9941-9189daf09ecc")
    }

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
    }

    steps {
        script {
            name = "Deploy to PyPi"
            id = "Deploy_to_PyPi"
            workingDir = "imod-python"
            scriptContent = """
                pixi run --frozen twine check ../dist/*
                pixi run --frozen twine upload ../dist/*
            """.trimIndent()
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=4 --memory=16g"""
            dockerPull = false
        }
    }

    dependencies {
        dependency(BuildPackage) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = "+:dist.zip!** => dist"
            }
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})

object DeployPages : BuildType({
    name = "Deploy Pages"

    enablePersonalBuilds = false
    type = Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.SUCCESSFUL

    params {
        param("GH_USER", "deltares-service-account")
        param("env.GH_TOKEN", "%github_deltares-service-account_access_token%")
    }

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
        showDependenciesChanges = true
    }

    steps {
        script {
            name = "Deploy Pages"
            workingDir = "imod-python"
            scriptContent = """
                echo on
                echo "Setup git"
                git config --global user.email "svc-teamcity-ansible@DIRECTORY.INTRA"
                git config --global user.name "SVC TeamCity Ansible"

                echo "Checkout imod-python-pages"
                git remote set-url origin https://%GH_USER%:%env.GH_TOKEN%@github.com/Deltares/imod-python.git
                git fetch origin gh-pages
                git switch gh-pages
                
                echo "Clear imod-python-pages"
                git rm -r *
                git clean -fdx
                
                echo "Copy documentation"
                xcopy %system.teamcity.build.checkoutDir%\docs . /e /h /i
                type nul > .nojekyll
                
                echo "Commit documentation"
                git add -A
                git commit -m "Published documentation for branch/tag %teamcity.build.branch%"
                
                echo "Push documentation"
                git push origin gh-pages
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=4 --memory=16g"""
            dockerPull = false
        }
    }

    dependencies {
        dependency(BuildPages) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
                synchronizeRevisions = false
            }

            artifacts {
                artifactRules = "+:documentation.zip!** => docs"
            }
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})

object DeployAll : BuildType({
    name = "Deploy All"

    enablePersonalBuilds = false
    type = Type.COMPOSITE
    maxRunningBuilds = 1

    vcs {
        root(AbsoluteId("iMOD6_IMODPython_ImodPython"), "+:. => imod-python")

        cleanCheckout = true
        branchFilter = """
            +:*
            -:<default>
            -:refs/heads/gh-pages
        """.trimIndent()
        showDependenciesChanges = true
    }

    dependencies {
        snapshot(CreateGitHubRelease) {
        }
        snapshot(DeployPackage) {
        }
        snapshot(DeployPages) {
        }
    }
})