package Pixi

import jetbrains.buildServer.configs.kotlin.AbsoluteId
import jetbrains.buildServer.configs.kotlin.BuildType
import jetbrains.buildServer.configs.kotlin.Project
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.triggers.schedule

object PixiProject : Project({
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
                    pixi run -e pixi-update update
                    
                    echo "Add any changes" 
                    git add pixi.lock
                    
                    if (git status -suno pixi.lock) 
                    {
                      echo "Setup username and email"
                      git config --global user.name "Teamcity"
                      git config --global user.email "teamcity@deltares.nl"
                      
                      echo "Commit changes"
                      git commit -m "Update pixi.lock"

                      echo "Push changes" 
                      git push -u origin pixi_update_%build.counter%

                      echo "Format PR body"
                      ${'$'}diff = Get-Content -Path diff.md
                      Set-Content body.md 'Teamcity automatically updated the dependencies defined the pixi.toml file. Please verify that all tests succeed before merging'
                      Add-Content -Path body.md -Value "`r`n"
                      Add-Content -Path body.md -Value ${'$'}diff
                      
                      echo "Create PR"
                      pixi run --environment default gh pr create --title "[TEAMCITY] Update project dependencies" --body-file body.md --reviewer JoerivanEngelen,Manangka
                      echo "Changes pushed and PR created"
                    }
                    else
                    {
                      echo "No changes found"
                    }
                """.trimIndent()
            }
            noProfile = false
            param("plugin.docker.imagePlatform", "windows")
            param("plugin.docker.pull.enabled", "false")
            param("plugin.docker.imageId", "%DockerContainer%:%DockerVersion%")
            param("plugin.docker.run.parameters", "--cpus=4 --memory=16g")
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

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_134"
            }
        }
    }
})