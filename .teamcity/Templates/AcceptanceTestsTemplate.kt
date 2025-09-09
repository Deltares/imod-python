package Templates

import jetbrains.buildServer.configs.kotlin.DslContext
import jetbrains.buildServer.configs.kotlin.Template
import jetbrains.buildServer.configs.kotlin.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object AcceptanceTestsTemplate : Template({
    name = "AcceptanceTestsTemplate"

    allowExternalStatus = true

    vcs {
        root(DslContext.settingsRoot, "+:. => imod-python")
        cleanCheckout = true
    }


    params {
        password("env.access_key", "credentialsJSON:2978988a-493b-44ed-9fcb-6cd6d2c2c673")
        password("env.secret_access_key", "credentialsJSON:409a55c0-b2e7-438c-98dd-f0404b83cabb")
    }


    steps {
        script {
            name = "Run acceptance tests"
            id = "Run_acceptance_tests"
            workingDir = "imod-python"
            scriptContent = """
                SET PATH=%%PATH%%;%system.teamcity.build.checkoutDir%\modflow6
                
                pixi run --environment user-acceptance --frozen dvc remote modify --local minio access_key_id %env.access_key%
                pixi run --environment user-acceptance --frozen dvc remote modify --local minio secret_access_key %env.secret_access_key%
                
                pixi run --environment user-acceptance --frozen user_acceptance
            """.trimIndent()
            formatStderrAsError = true
            dockerImage = "%DockerContainer%:%DockerVersion%"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Windows
            dockerRunParameters = """--cpus=8 --memory=32g"""
            dockerPull = false
        }
    }

})