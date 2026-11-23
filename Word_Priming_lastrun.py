#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on November 15, 2026, at 21:23
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.4'
expName = 'Word_Priming'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [960, 600]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data' + os.sep + '%s_%s' % (expInfo['participant'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\ribha\\Codes\\dhriti_lab_report\\Lab_5\\Word_Priming_PsychoPy\\Word_Priming_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color='black', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instructions" ---
    introText = visual.TextStim(win=win, name='introText',
        text='WORD PRIMING EXPERIMENT\n\nSTUDY PHASE\n\nYou will see 15 words, one at a time.\nTry to remember each word.\nRate your familiarity with each word from 1 to 5.\nEach word remains on screen for 8 seconds.\n\nPress SPACE to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.036, wrapWidth=0.99, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    startStudy = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "StudyPhase" ---
    studyHint = visual.TextStim(win=win, name='studyHint',
        text='STUDY PHASE\nRemember the word and choose a rating (1 to 5).',
        font='Arial',
        pos=(0, 0.36), draggable=False, height=0.026, wrapWidth=0.95, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    StudyText = visual.TextStim(win=win, name='StudyText',
        text='',
        font='Arial',
        pos=(0, 0.11), draggable=False, height=0.100, wrapWidth=0.9, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    familiarityLabel = visual.TextStim(win=win, name='familiarityLabel',
        text='Rate how familiar the word is:',
        font='Arial',
        pos=(0, -0.06), draggable=False, height=0.030, wrapWidth=0.90, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Studyslider = visual.Slider(win=win, name='Studyslider',
        startValue=None, size=(0.78, 0.055), pos=(0, -0.245), units=win.units,
        labels=['Familiar', 'Unfamiliar'],ticks=None, granularity=1,
        style='radio', styleTweaks=[], opacity=None,
        labelColor='white', markerColor='lightgrey', lineColor='white', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.027,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "TestInstructions" ---
    testInstructions = visual.TextStim(win=win, name='testInstructions',
        text='TEST PHASE\n\nComplete each incomplete word.\nType the FULL word in the box.\nClick SUBMIT to continue to the next word.\nThe word clue is visible for up to 10 seconds.\nYou can still submit your answer afterwards.\n\nPress SPACE to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.034, wrapWidth=0.98, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    startTest = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "TestPhase" ---
    testHint = visual.TextStim(win=win, name='testHint',
        text='Enter the full word, then click SUBMIT.',
        font='Arial',
        pos=(0, 0.355), draggable=False, height=0.028, wrapWidth=0.9, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    TestText = visual.TextStim(win=win, name='TestText',
        text='',
        font='Arial',
        pos=(0, 0.20), draggable=False, height=0.105, wrapWidth=0.9, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    test_textbox = visual.TextBox2(
         win, text=None, placeholder='Type the full word here', font='Arial',
         ori=0.0, pos=(0, -0.10), draggable=False,      letterHeight=0.037,
         size=(0.82, 0.105), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.01, alignment='center',
         anchor='center', overflow='visible',
         fillColor='#202020', borderColor='white',
         flipHoriz=False, flipVert=False, 
         languageStyle='LTR',
         formattingSyntax='md',
         editable=True,
         name='test_textbox',
         depth=-2, autoLog=True,
    )
    submit_button = visual.Rect(
        win=win, name='submit_button',
        width=(0.55, 0.105)[0], height=(0.55, 0.105)[1],
        ori=0.0, pos=(0, -0.325), draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor='#343434',
        opacity=1.0, depth=-3.0, interpolate=True)
    submit_text = visual.TextStim(win=win, name='submit_text',
        text='SUBMIT',
        font='Arial',
        pos=(0, -0.325), draggable=False, height=0.037, wrapWidth=0.60, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    testmouse = event.Mouse(win=win)
    x, y = [None, None]
    testmouse.mouseClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instructions" ---
    # create an object to store info about Routine Instructions
    Instructions = data.Routine(
        name='Instructions',
        components=[introText, startStudy],
    )
    Instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for startStudy
    startStudy.keys = []
    startStudy.rt = []
    _startStudy_allKeys = []
    # store start times for Instructions
    Instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions.tStart = globalClock.getTime(format='float')
    Instructions.status = STARTED
    thisExp.addData('Instructions.started', Instructions.tStart)
    Instructions.maxDuration = None
    # keep track of which components have finished
    InstructionsComponents = Instructions.components
    for thisComponent in Instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    thisExp.currentRoutine = Instructions
    Instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *introText* updates
        
        # if introText is starting this frame...
        if introText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            introText.frameNStart = frameN  # exact frame index
            introText.tStart = t  # local t and not account for scr refresh
            introText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(introText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'introText.started')
            # update status
            introText.status = STARTED
            introText.setAutoDraw(True)
        
        # if introText is active this frame...
        if introText.status == STARTED:
            # update params
            pass
        
        # *startStudy* updates
        waitOnFlip = False
        
        # if startStudy is starting this frame...
        if startStudy.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            startStudy.frameNStart = frameN  # exact frame index
            startStudy.tStart = t  # local t and not account for scr refresh
            startStudy.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(startStudy, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'startStudy.started')
            # update status
            startStudy.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(startStudy.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(startStudy.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if startStudy.status == STARTED and not waitOnFlip:
            theseKeys = startStudy.getKeys(keyList=["space"], ignoreKeys=["escape"], waitRelease=False)
            _startStudy_allKeys.extend(theseKeys)
            if len(_startStudy_allKeys):
                startStudy.keys = _startStudy_allKeys[-1].name  # just the last key pressed
                startStudy.rt = _startStudy_allKeys[-1].rt
                startStudy.duration = _startStudy_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in Instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions
    Instructions.tStop = globalClock.getTime(format='float')
    Instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions.stopped', Instructions.tStop)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('study_conditions.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "StudyPhase" ---
        # create an object to store info about Routine StudyPhase
        StudyPhase = data.Routine(
            name='StudyPhase',
            components=[studyHint, StudyText, familiarityLabel, Studyslider],
        )
        StudyPhase.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        StudyText.setText(Study_word)
        Studyslider.reset()
        # store start times for StudyPhase
        StudyPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        StudyPhase.tStart = globalClock.getTime(format='float')
        StudyPhase.status = STARTED
        thisExp.addData('StudyPhase.started', StudyPhase.tStart)
        StudyPhase.maxDuration = 8.0
        # keep track of which components have finished
        StudyPhaseComponents = StudyPhase.components
        for thisComponent in StudyPhase.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "StudyPhase" ---
        thisExp.currentRoutine = StudyPhase
        StudyPhase.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 8.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > StudyPhase.maxDuration-frameTolerance:
                StudyPhase.maxDurationReached = True
                continueRoutine = False
            
            # *studyHint* updates
            
            # if studyHint is starting this frame...
            if studyHint.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                studyHint.frameNStart = frameN  # exact frame index
                studyHint.tStart = t  # local t and not account for scr refresh
                studyHint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(studyHint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'studyHint.started')
                # update status
                studyHint.status = STARTED
                studyHint.setAutoDraw(True)
            
            # if studyHint is active this frame...
            if studyHint.status == STARTED:
                # update params
                pass
            
            # if studyHint is stopping this frame...
            if studyHint.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > studyHint.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    studyHint.tStop = t  # not accounting for scr refresh
                    studyHint.tStopRefresh = tThisFlipGlobal  # on global time
                    studyHint.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'studyHint.stopped')
                    # update status
                    studyHint.status = FINISHED
                    studyHint.setAutoDraw(False)
            
            # *StudyText* updates
            
            # if StudyText is starting this frame...
            if StudyText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                StudyText.frameNStart = frameN  # exact frame index
                StudyText.tStart = t  # local t and not account for scr refresh
                StudyText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(StudyText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'StudyText.started')
                # update status
                StudyText.status = STARTED
                StudyText.setAutoDraw(True)
            
            # if StudyText is active this frame...
            if StudyText.status == STARTED:
                # update params
                pass
            
            # if StudyText is stopping this frame...
            if StudyText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > StudyText.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    StudyText.tStop = t  # not accounting for scr refresh
                    StudyText.tStopRefresh = tThisFlipGlobal  # on global time
                    StudyText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'StudyText.stopped')
                    # update status
                    StudyText.status = FINISHED
                    StudyText.setAutoDraw(False)
            
            # *familiarityLabel* updates
            
            # if familiarityLabel is starting this frame...
            if familiarityLabel.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                familiarityLabel.frameNStart = frameN  # exact frame index
                familiarityLabel.tStart = t  # local t and not account for scr refresh
                familiarityLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(familiarityLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'familiarityLabel.started')
                # update status
                familiarityLabel.status = STARTED
                familiarityLabel.setAutoDraw(True)
            
            # if familiarityLabel is active this frame...
            if familiarityLabel.status == STARTED:
                # update params
                pass
            
            # if familiarityLabel is stopping this frame...
            if familiarityLabel.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > familiarityLabel.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    familiarityLabel.tStop = t  # not accounting for scr refresh
                    familiarityLabel.tStopRefresh = tThisFlipGlobal  # on global time
                    familiarityLabel.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'familiarityLabel.stopped')
                    # update status
                    familiarityLabel.status = FINISHED
                    familiarityLabel.setAutoDraw(False)
            
            # *Studyslider* updates
            
            # if Studyslider is starting this frame...
            if Studyslider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Studyslider.frameNStart = frameN  # exact frame index
                Studyslider.tStart = t  # local t and not account for scr refresh
                Studyslider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Studyslider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Studyslider.started')
                # update status
                Studyslider.status = STARTED
                Studyslider.setAutoDraw(True)
            
            # if Studyslider is active this frame...
            if Studyslider.status == STARTED:
                # update params
                pass
            
            # if Studyslider is stopping this frame...
            if Studyslider.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Studyslider.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Studyslider.tStop = t  # not accounting for scr refresh
                    Studyslider.tStopRefresh = tThisFlipGlobal  # on global time
                    Studyslider.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Studyslider.stopped')
                    # update status
                    Studyslider.status = FINISHED
                    Studyslider.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=StudyPhase,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                StudyPhase.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if StudyPhase.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in StudyPhase.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "StudyPhase" ---
        for thisComponent in StudyPhase.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for StudyPhase
        StudyPhase.tStop = globalClock.getTime(format='float')
        StudyPhase.tStopRefresh = tThisFlipGlobal
        thisExp.addData('StudyPhase.stopped', StudyPhase.tStop)
        trials.addData('Studyslider.response', Studyslider.getRating())
        trials.addData('Studyslider.rt', Studyslider.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if StudyPhase.maxDurationReached:
            routineTimer.addTime(-StudyPhase.maxDuration)
        elif StudyPhase.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-8.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "TestInstructions" ---
    # create an object to store info about Routine TestInstructions
    TestInstructions = data.Routine(
        name='TestInstructions',
        components=[testInstructions, startTest],
    )
    TestInstructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for startTest
    startTest.keys = []
    startTest.rt = []
    _startTest_allKeys = []
    # store start times for TestInstructions
    TestInstructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    TestInstructions.tStart = globalClock.getTime(format='float')
    TestInstructions.status = STARTED
    thisExp.addData('TestInstructions.started', TestInstructions.tStart)
    TestInstructions.maxDuration = None
    # keep track of which components have finished
    TestInstructionsComponents = TestInstructions.components
    for thisComponent in TestInstructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "TestInstructions" ---
    thisExp.currentRoutine = TestInstructions
    TestInstructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *testInstructions* updates
        
        # if testInstructions is starting this frame...
        if testInstructions.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            testInstructions.frameNStart = frameN  # exact frame index
            testInstructions.tStart = t  # local t and not account for scr refresh
            testInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(testInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'testInstructions.started')
            # update status
            testInstructions.status = STARTED
            testInstructions.setAutoDraw(True)
        
        # if testInstructions is active this frame...
        if testInstructions.status == STARTED:
            # update params
            pass
        
        # *startTest* updates
        waitOnFlip = False
        
        # if startTest is starting this frame...
        if startTest.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            startTest.frameNStart = frameN  # exact frame index
            startTest.tStart = t  # local t and not account for scr refresh
            startTest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(startTest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'startTest.started')
            # update status
            startTest.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(startTest.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(startTest.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if startTest.status == STARTED and not waitOnFlip:
            theseKeys = startTest.getKeys(keyList=["space"], ignoreKeys=["escape"], waitRelease=False)
            _startTest_allKeys.extend(theseKeys)
            if len(_startTest_allKeys):
                startTest.keys = _startTest_allKeys[-1].name  # just the last key pressed
                startTest.rt = _startTest_allKeys[-1].rt
                startTest.duration = _startTest_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=TestInstructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            TestInstructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if TestInstructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in TestInstructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "TestInstructions" ---
    for thisComponent in TestInstructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for TestInstructions
    TestInstructions.tStop = globalClock.getTime(format='float')
    TestInstructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('TestInstructions.stopped', TestInstructions.tStop)
    thisExp.nextEntry()
    # the Routine "TestInstructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('test_conditions.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        trials_2.status = STARTED
        if hasattr(thisTrial_2, 'status'):
            thisTrial_2.status = STARTED
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "TestPhase" ---
        # create an object to store info about Routine TestPhase
        TestPhase = data.Routine(
            name='TestPhase',
            components=[testHint, TestText, test_textbox, submit_button, submit_text, testmouse],
        )
        TestPhase.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        TestText.setText(test_word)
        test_textbox.reset()
        # setup some python lists for storing info about the testmouse
        testmouse.x = []
        testmouse.y = []
        testmouse.leftButton = []
        testmouse.midButton = []
        testmouse.rightButton = []
        testmouse.time = []
        testmouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from textboxFocus
        test_textbox.hasFocus = True
        # store start times for TestPhase
        TestPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TestPhase.tStart = globalClock.getTime(format='float')
        TestPhase.status = STARTED
        thisExp.addData('TestPhase.started', TestPhase.tStart)
        TestPhase.maxDuration = None
        # keep track of which components have finished
        TestPhaseComponents = TestPhase.components
        for thisComponent in TestPhase.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TestPhase" ---
        thisExp.currentRoutine = TestPhase
        TestPhase.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_2, 'status') and thisTrial_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *testHint* updates
            
            # if testHint is starting this frame...
            if testHint.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                testHint.frameNStart = frameN  # exact frame index
                testHint.tStart = t  # local t and not account for scr refresh
                testHint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testHint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'testHint.started')
                # update status
                testHint.status = STARTED
                testHint.setAutoDraw(True)
            
            # if testHint is active this frame...
            if testHint.status == STARTED:
                # update params
                pass
            
            # *TestText* updates
            
            # if TestText is starting this frame...
            if TestText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                TestText.frameNStart = frameN  # exact frame index
                TestText.tStart = t  # local t and not account for scr refresh
                TestText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(TestText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'TestText.started')
                # update status
                TestText.status = STARTED
                TestText.setAutoDraw(True)
            
            # if TestText is active this frame...
            if TestText.status == STARTED:
                # update params
                pass
            
            # if TestText is stopping this frame...
            if TestText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > TestText.tStartRefresh + 10.0-frameTolerance:
                    # keep track of stop time/frame for later
                    TestText.tStop = t  # not accounting for scr refresh
                    TestText.tStopRefresh = tThisFlipGlobal  # on global time
                    TestText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'TestText.stopped')
                    # update status
                    TestText.status = FINISHED
                    TestText.setAutoDraw(False)
            
            # *test_textbox* updates
            
            # if test_textbox is starting this frame...
            if test_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_textbox.frameNStart = frameN  # exact frame index
                test_textbox.tStart = t  # local t and not account for scr refresh
                test_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_textbox.started')
                # update status
                test_textbox.status = STARTED
                test_textbox.setAutoDraw(True)
            
            # if test_textbox is active this frame...
            if test_textbox.status == STARTED:
                # update params
                pass
            
            # *submit_button* updates
            
            # if submit_button is starting this frame...
            if submit_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                submit_button.frameNStart = frameN  # exact frame index
                submit_button.tStart = t  # local t and not account for scr refresh
                submit_button.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(submit_button, 'tStartRefresh')  # time at next scr refresh
                # update status
                submit_button.status = STARTED
                submit_button.setAutoDraw(True)
            
            # if submit_button is active this frame...
            if submit_button.status == STARTED:
                # update params
                pass
            
            # *submit_text* updates
            
            # if submit_text is starting this frame...
            if submit_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                submit_text.frameNStart = frameN  # exact frame index
                submit_text.tStart = t  # local t and not account for scr refresh
                submit_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(submit_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'submit_text.started')
                # update status
                submit_text.status = STARTED
                submit_text.setAutoDraw(True)
            
            # if submit_text is active this frame...
            if submit_text.status == STARTED:
                # update params
                pass
            # *testmouse* updates
            
            # if testmouse is starting this frame...
            if testmouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                testmouse.frameNStart = frameN  # exact frame index
                testmouse.tStart = t  # local t and not account for scr refresh
                testmouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(testmouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('testmouse.started', t)
                # update status
                testmouse.status = STARTED
                testmouse.mouseClock.reset()
                # if button is down already this ISN'T a new click
                testmouse.prevButtonState = testmouse.getPressed()
            if testmouse.status == STARTED:  # only update if started and not finished!
                buttons = testmouse.getPressed()
                # button state changed?
                if buttons != testmouse.prevButtonState:
                    testmouse.prevButtonState = buttons
                    # state changed to a new click
                    if sum(buttons) > 0:
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(submit_button, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(testmouse):
                                gotValidClick = True
                                testmouse.clicked_name.append(obj.name)
                                testmouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = testmouse.getPos()
                            testmouse.x.append(float(x))
                            testmouse.y.append(float(y))
                            buttons = testmouse.getPressed()
                            testmouse.leftButton.append(buttons[0])
                            testmouse.midButton.append(buttons[1])
                            testmouse.rightButton.append(buttons[2])
                            testmouse.time.append(testmouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=TestPhase,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                TestPhase.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if TestPhase.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in TestPhase.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TestPhase" ---
        for thisComponent in TestPhase.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TestPhase
        TestPhase.tStop = globalClock.getTime(format='float')
        TestPhase.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TestPhase.stopped', TestPhase.tStop)
        trials_2.addData('test_textbox.text',test_textbox.text)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('testmouse.x', testmouse.x)
        trials_2.addData('testmouse.y', testmouse.y)
        trials_2.addData('testmouse.leftButton', testmouse.leftButton)
        trials_2.addData('testmouse.midButton', testmouse.midButton)
        trials_2.addData('testmouse.rightButton', testmouse.rightButton)
        trials_2.addData('testmouse.time', testmouse.time)
        trials_2.addData('testmouse.clicked_name', testmouse.clicked_name)
        # the Routine "TestPhase" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial_2 as finished
        if hasattr(thisTrial_2, 'status'):
            thisTrial_2.status = FINISHED
        # if awaiting a pause, pause now
        if trials_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    trials_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
